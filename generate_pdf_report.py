from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem
from reportlab.lib.units import inch
from pathlib import Path
import re

PDF_PATH = Path('code_overview_report.pdf')
MD_PATH = Path('code_overview_report.md')

styles = getSampleStyleSheet()
heading1 = styles['Heading1']
heading1.spaceAfter = 12
heading2 = styles['Heading2']
heading2.spaceAfter = 10
body = styles['BodyText']
body.spaceAfter = 8
styles.add(ParagraphStyle(name='List', parent=styles['BodyText'], leftIndent=18, spaceAfter=4))


def parse_markdown(md_text):
    elements = []
    lines = md_text.splitlines()
    list_acc = []

    def flush_list():
        nonlocal list_acc
        if list_acc:
            items = [ListItem(Paragraph(item, styles['BodyText']), leftIndent=10) for item in list_acc]
            elements.append(ListFlowable(items, bulletType='bullet', leftIndent=18))
            elements.append(Spacer(1, 8))
            list_acc = []

    for line in lines:
        line = line.rstrip()
        if not line:
            flush_list()
            continue

        if line.startswith('# '):
            flush_list()
            elements.append(Paragraph(line[2:].strip(), styles['Heading1']))
            continue
        if line.startswith('## '):
            flush_list()
            elements.append(Paragraph(line[3:].strip(), styles['Heading2']))
            continue
        if line.startswith('- '):
            list_acc.append(line[2:].strip())
            continue
        if line.startswith('```'):
            continue
        if line.startswith('> '):
            flush_list()
            elements.append(Paragraph(line[2:].strip(), styles['BodyText']))
            continue
        if line.startswith('!['):
            flush_list()
            m = re.match(r'!\[.*?\]\((.*?)\)', line)
            if m:
                img_path = Path(m.group(1))
                if img_path.exists():
                    img = Image(str(img_path), width=6.5*inch, height=3.5*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 12))
                else:
                    elements.append(Paragraph(f'Image not found: {img_path}', styles['BodyText']))
            continue

        # paragraphs and inline code blocks
        text = line.replace('`', '')
        elements.append(Paragraph(text, styles['BodyText']))

    flush_list()
    return elements


def main():
    if not MD_PATH.exists():
        raise FileNotFoundError(f'Markdown file not found: {MD_PATH}')
    md_text = MD_PATH.read_text(encoding='utf-8')
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=letter,
                            leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    story = parse_markdown(md_text)
    doc.build(story)
    print(f'PDF generated: {PDF_PATH}')


if __name__ == '__main__':
    main()
