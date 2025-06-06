import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from pdf2image import convert_from_bytes
from datetime import datetime, timedelta
from pathlib import Path
import pikepdf
import os
import re
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from typing import Optional

# --------------------------------------------
# App Configuration
APP_NAME = "DigiHooshHoma SABA (By AMK)"
APP_ICON = "images/homa_logo.png"
SECURED_FOLDER = "secured"

# DPI options for text→image conversion
DPI_OPTIONS = {
    "Low (150 DPI)":   150,
    "Medium (300 DPI)": 300,
    "High (600 DPI)": 600,
    "Ultra (1200 DPI)": 1200,
    "Custom-1 (50 DPI)":  50,
    "Custom-2 (75 DPI)":  75,
    "Custom-3 (100 DPI)": 100,
    "Custom-4 (200 DPI)": 200,
    "Custom-5 (250 DPI)": 250,
}

# Standard paper sizes (mm), including local Iranian formats
STANDARD_PAPERS = {
    "A0":       (841, 1189),
    "A1":       (594, 841),
    "A2":       (420, 594),
    "A3":       (297, 420),
    "A4":       (210, 297),
    "A5":       (148, 210),
    "B5":       (176, 250),
    "Letter":   (216, 279),
    "Legal":    (216, 356),
    "خشتی":        (210, 210),
    "۱۴/۵ × ۱۴/۵": (145, 145),
    "پالتویی":     (120, 205),
    "جیبی":        (120, 170),
    "رحلی":        (205, 285),
    "وزیری":       (170, 240),
    "رقعی":        (145, 210),
}

def points_to_mm(pt: float) -> float:
    return pt * 25.4 / 72.0

def match_standard_paper(width_mm: float, height_mm: float, tol_mm: float = 2.0) -> str:
    for name, (w_std, h_std) in STANDARD_PAPERS.items():
        if abs(width_mm - w_std) <= tol_mm and abs(height_mm - h_std) <= tol_mm:
            return name
        if abs(width_mm - h_std) <= tol_mm and abs(height_mm - w_std) <= tol_mm:
            return name
    return "Custom/Unknown"

def get_page_size_info(pdf_bytes: bytes) -> dict:
    reader = PdfReader(BytesIO(pdf_bytes))
    page = reader.pages[0]
    w_pt = float(page.mediabox.width)
    h_pt = float(page.mediabox.height)
    w_mm = points_to_mm(w_pt)
    h_mm = points_to_mm(h_pt)
    paper_name = match_standard_paper(w_mm, h_mm, tol_mm=2.0)
    return {
        "width_pt": round(w_pt, 2),
        "height_pt": round(h_pt, 2),
        "width_mm": round(w_mm, 1),
        "height_mm": round(h_mm, 1),
        "paper_format": paper_name
    }

def parse_timezone_offset(offset_str: str):
    match = re.match(r'^([+-]?)(\d{1,2}):(\d{2})$', offset_str)
    if not match:
        return None
    sign, hours, minutes = match.groups()
    delta = timedelta(hours=int(hours), minutes=int(minutes))
    return -delta if sign == '-' else delta

def expand_outlines(item, current_level=0, max_level=1):
    if not item or current_level > max_level:
        return
    if '/First' in item:
        item['/Count'] = abs(item.get('/Count', 0))
        child = item['/First']
        while child is not None:
            expand_outlines(child, current_level + 1, max_level)
            child = child.get('/Next')

def convert_pdf_to_image_noreenc(pdf_bytes: bytes, dpi: int) -> bytes:
    reader = PdfReader(BytesIO(pdf_bytes))
    page_sizes = [(float(page.mediabox.width), float(page.mediabox.height)) for page in reader.pages]
    images = convert_from_bytes(pdf_bytes, dpi=dpi)

    with BytesIO() as buf:
        c = canvas.Canvas(buf, pagesize=(1, 1))
        for idx, img in enumerate(images):
            w_pt, h_pt = page_sizes[idx]
            c.setPageSize((w_pt, h_pt))
            img_byte = BytesIO()
            img.save(img_byte, format='PNG')
            img_byte.seek(0)
            rep_image = ImageReader(img_byte)
            c.drawImage(rep_image, 0, 0, width=w_pt, height=h_pt)
            c.showPage()
        c.save()
        return buf.getvalue()

def extract_outline_tree(orig_pdf: pikepdf.Pdf) -> list:
    """
    Extracts the outline entries from orig_pdf.Root.Outlines into a list of dicts
    [{'title': str, 'page_index': int, 'children': [...]}, ...].
    Ensures only primitive data types are stored in the outline tree.
    """
    def _walk(outline_ref):
        items = []
        node = outline_ref.get('/First')
        while node is not None:
            title = node.get('/Title', '')
            dest = node.get('/Dest')
            page_index = 0
            if isinstance(dest, pikepdf.Array) and len(dest) > 0:
                page_ref = dest[0]
                for idx, p in enumerate(orig_pdf.pages):
                    if p.objgen == page_ref.objgen:
                        page_index = idx
                        break
            children = []
            if '/First' in node:
                children = _walk(node)
            items.append({
                'title': str(title),  # Ensure title is a string
                'page_index': int(page_index),  # Ensure page_index is an integer
                'children': children
            })
            node = node.get('/Next')
        return items

    if '/Outlines' not in orig_pdf.Root:
        return []
    return _walk(orig_pdf.Root.Outlines)

def apply_outline_tree(new_pdf: pikepdf.Pdf, tree: list):
    """
    Creates a new /Outlines structure in new_pdf using pikepdf.Outline.
    """
    def _build(nodes, parent_outline):
        last_item = None
        for node in nodes:
            # Always use the page from new_pdf, not from orig_pdf
            page = new_pdf.pages[node['page_index']].obj
            title = node['title']
            
            # Create the outline item
            item = pikepdf.Dictionary({
                '/Title': title,
                '/Dest': [page, pikepdf.Name('/Fit')],
                '/Parent': parent_outline,
                '/Next': None,
                '/Prev': None,
                '/First': None,
                '/Last': None,
                '/Count': 0
            })
            
            # Link items together
            if last_item is None:
                parent_outline['/First'] = item
            else:
                last_item['/Next'] = item
                item['/Prev'] = last_item
            
            last_item = item
            parent_outline['/Last'] = item
            
            # Process children if any exist
            if node['children']:
                _build(node['children'], item)

    # Create outline dictionary if it doesn't exist
    if '/Outlines' not in new_pdf.Root:
        new_pdf.Root['/Outlines'] = pikepdf.Dictionary({
            '/Type': '/Outlines',
            '/First': None,
            '/Last': None,
            '/Count': 0
        })

    outlines = new_pdf.Root['/Outlines']
    if tree:  # Only process if we have outline items
        _build(tree, outlines)

def secure_pdf(
    working_bytes: bytes,
    owner_pwd: str,
    user_pwd: str,
    permissions: Optional[dict] = None,
    encryption_method: str = "AES-256",
) -> bytes:
    """
    Encrypt, ensuring accessibility=False when content_copy=False.
    """
    try:
        pdf = pikepdf.Pdf.open(BytesIO(working_bytes))

        # Encryption config (disable accessibility if content_copy is False)
        permissions_obj = pikepdf.Permissions(
            print_lowres       = permissions.get('print_lowres', False),
            print_highres      = permissions.get('print_highres', False),
            modify_annotation  = permissions.get('modify_annotations', False),
            modify_form        = permissions.get('fill_forms', False),
            modify_other       = permissions.get('modify_document', False),
            modify_assembly    = permissions.get('assemble', False),
            extract            = permissions.get('content_copy', False),
            accessibility      = permissions.get('content_copy', False)
        )

        if encryption_method == "AES-256":
            encryption_obj = pikepdf.Encryption(
                owner=owner_pwd,
                user=user_pwd,
                allow=permissions_obj,
                R=6
            )
        elif encryption_method == "AES-128":
            encryption_obj = pikepdf.Encryption(
                owner=owner_pwd,
                user=user_pwd,
                allow=permissions_obj,
                R=5
            )
        elif encryption_method == "RC4-128":
            encryption_obj = pikepdf.Encryption(
                owner=owner_pwd,
                user=user_pwd,
                allow=permissions_obj,
                R=4
            )
        elif encryption_method == "RC4-40":
            encryption_obj = pikepdf.Encryption(
                owner=owner_pwd,
                user=user_pwd,
                allow=permissions_obj,
                R=3
            )
        else:
            encryption_obj = pikepdf.Encryption(
                owner=owner_pwd,
                user=user_pwd,
                allow=permissions_obj
            )

        out_buf = BytesIO()
        pdf.save(out_buf, encryption=encryption_obj)
        return out_buf.getvalue()

    except Exception as e:
        import traceback
        st.error(f"PDF Processing Error: {str(e)}")
        st.error(traceback.format_exc())
        return b""

def apply_outlines(pdf_bytes: bytes, max_size_mb: int = 50) -> bytes:
    """Applies outlines to the PDF with size check and timeout."""
    try:
        # Check file size first to prevent memory issues
        if len(pdf_bytes) > max_size_mb * 1024 * 1024:  # Convert MB to bytes
            st.warning(f"PDF too large (>{max_size_mb}MB) for outline processing. Skipping outlines.")
            return pdf_bytes

        pdf = pikepdf.open(BytesIO(pdf_bytes))
        
        # Only process if outlines exist in original
        if '/Outlines' not in pdf.Root:
            return pdf_bytes
            
        tree = extract_outline_tree(pdf)
        if tree:
            apply_outline_tree(pdf, tree)
        
        out_buf = BytesIO()
        pdf.save(out_buf)
        return out_buf.getvalue()

    except Exception as e:
        st.warning(f"Outline processing skipped: {str(e)}")
        return pdf_bytes  # Return original bytes if outline fails

def save_to_folder(
    pdf_bytes: bytes,
    original_name: str,
    prefix: str
) -> str:
    """
    Save to SECURED_FOLDER with filename format:
      - normal protection:     s4p_TIMESTAMP.pdf
      - image protection:      is4p_TIMESTAMP.pdf
    """
    os.makedirs(SECURED_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(original_name).stem

    output_name = f"{base}_{prefix}{timestamp}.pdf"  # <-- Added underscore for clarity
    output_path = os.path.join(SECURED_FOLDER, output_name)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    return output_path

# --------------------------------------------
# Streamlit UI
st.set_page_config(page_title=APP_NAME, page_icon="🔒", layout="wide")
st.title(f"🔒 {APP_NAME}")

def display_logo():
    try:
        if LOGO_PATH.lower().endswith('.mp4'):
            st.sidebar.video(LOGO_PATH, format='video/mp4')
        else:
            st.sidebar.image(LOGO_PATH, width=200)
    except:
        st.sidebar.warning("Logo not found")

LOGO_PATH = "images/homa_logo.png"
display_logo()

st.sidebar.markdown(f"### {APP_NAME}")
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Form"):
    st.rerun()

uploaded = st.file_uploader("Upload PDF", type=['pdf'])
if uploaded:
    original_bytes = uploaded.read()
    input_info = None
    try:
        pdf_reader = PdfReader(BytesIO(original_bytes))
        st.session_state.pdf_metadata = dict(pdf_reader.metadata or {})
        input_info = get_page_size_info(original_bytes)
        st.sidebar.subheader("Input PDF Info")
        st.sidebar.write(f"• Size (points): {input_info['width_pt']} pt × {input_info['height_pt']} pt")
        st.sidebar.write(f"• Size (mm): {input_info['width_mm']} mm × {input_info['height_mm']} mm")
        st.sidebar.write(f"• Matched format: {input_info['paper_format']}")
        st.sidebar.write(f"• Total pages: {len(pdf_reader.pages)}")
        if st.session_state.pdf_metadata:
            st.sidebar.write("Original Metadata:")
            for k, v in st.session_state.pdf_metadata.items():
                st.sidebar.write(f"- {k}: {v}")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        original_bytes = None
        input_info = None
else:
    original_bytes = None
    input_info = None

with st.form("pdf_form"):
    # Passwords
    col1, col2 = st.columns(2)
    with col1:
        owner_pwd = st.text_input("Owner Password*", type="password")
    with col2:
        user_pwd = st.text_input("User Password (allows content copy)", type="password")

    # Encryption method
    with st.expander("🔐 Encryption Options", expanded=True):
        encryption_method = st.selectbox(
            "Encryption Method",
            ["AES-256", "AES-128", "RC4-128", "RC4-40"],
            index=0
        )

    # Text Protection
    with st.expander("✍️ Text Protection", expanded=False):
        prevent_selection = st.checkbox("Prevent text selection (convert to image)", value=False)
        selected_dpi = st.selectbox("Image Quality (DPI)", list(DPI_OPTIONS.keys()), index=1)

    # Security Restrictions
    with st.expander("⚙️ Security Restrictions", expanded=False):
        st.info("Check boxes to ALLOW each permission")
        col1, col2 = st.columns(2)
        with col1:
            print_lowres       = st.checkbox("Allow Low-Res Printing", True)
            print_highres      = st.checkbox("Allow High-Res Printing", False)
            content_copy       = st.checkbox("Allow Content Copying", False)
            modify_annotations = st.checkbox("Allow Modify Annotations", False)
        with col2:
            fill_forms       = st.checkbox("Allow Fill Forms", False)
            modify_document  = st.checkbox("Allow Modify Document", False)
            assemble         = st.checkbox("Allow Assemble Document", False)

    submitted = st.form_submit_button("🛡️ Secure PDF")

if submitted and original_bytes and owner_pwd:
    with st.spinner("Applying security settings..."):
        # Create permissions dictionary from form inputs
        permissions = {
            'print_lowres': print_lowres,
            'print_highres': print_highres,
            'content_copy': content_copy,
            'modify_annotations': modify_annotations,
            'fill_forms': fill_forms,
            'modify_document': modify_document,
            'assemble': assemble
        }

        # 1) Secure PDF (Normal)
        secured_bytes = secure_pdf(
            working_bytes=original_bytes,
            owner_pwd=owner_pwd,
            user_pwd=user_pwd,
            permissions=permissions,
            encryption_method=encryption_method,
        )

        # Save the normally secured PDF first
        s4p_filename = save_to_folder(
            pdf_bytes=secured_bytes,
            original_name=uploaded.name if uploaded else "output.pdf",
            prefix="s4p_"
        )

        # 2) Process outlines only if file size is reasonable
        if len(secured_bytes) <= 50 * 1024 * 1024:  # 50MB limit
            try:
                os4p_bytes = apply_outlines(secured_bytes)
                if os4p_bytes != secured_bytes:  # Only save if outlines were actually added
                    os4p_filename = save_to_folder(
                        pdf_bytes=os4p_bytes,
                        original_name=uploaded.name if uploaded else "output.pdf",
                        prefix="os4p_"
                    )
                else:
                    os4p_filename = None
            except Exception:
                os4p_filename = None
        else:
            os4p_filename = None
            st.warning("PDF too large for outline processing")

        # 3) Text→image if needed
        if prevent_selection:
            with st.spinner("Converting to images..."):
                dpi_value = DPI_OPTIONS[selected_dpi]
                image_protected_bytes = convert_pdf_to_image_noreenc(original_bytes, dpi=dpi_value)
                
                is4p_bytes = secure_pdf(
                    working_bytes=image_protected_bytes,
                    owner_pwd=owner_pwd,
                    user_pwd=user_pwd,
                    permissions=permissions,
                    encryption_method=encryption_method,
                )

                is4p_filename = save_to_folder(
                    pdf_bytes=is4p_bytes,
                    original_name=uploaded.name if uploaded else "output.pdf",
                    prefix=f"is4p_{dpi_value}dpi_"
                )
                
                # Skip outline processing for image-protected PDFs
                ois4p_filename = None
        else:
            is4p_filename = None
            ois4p_filename = None

        st.balloons()
        st.success("PDF processing complete!")

        # Provide download links
        st.write("Download Links:")
        if os4p_filename:
            with open(os4p_filename, "rb") as file:
                st.download_button(
                    label=f"💾 Download Secure PDF with Outlines",
                    data=file,
                    file_name=Path(os4p_filename).name,
                    mime="application/pdf"
                )
        if s4p_filename:
            with open(s4p_filename, "rb") as file:
                st.download_button(
                    label=f"💾 Download Secure PDF (Normal)",
                    data=file,
                    file_name=Path(s4p_filename).name,
                    mime="application/pdf"
                )
        if is4p_filename:
            with open(is4p_filename, "rb") as file:
                st.download_button(
                    label=f"💾 Download Secure PDF (Image Protected)",
                    data=file,
                    file_name=Path(is4p_filename).name,
                    mime="application/pdf"
                )
        if ois4p_filename:
            with open(ois4p_filename, "rb") as file:
                st.download_button(
                    label=f"💾 Download Secure PDF with Outlines (Image Protected)",
                    data=file,
                    file_name=Path(ois4p_filename).name,
                    mime="application/pdf"
                )
