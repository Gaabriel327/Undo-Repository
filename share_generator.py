from PIL import Image, ImageDraw, ImageFont
import os

def generate_share_image(question, answer, username):
    width, height = 1080, 1920
    background_color = "#ffffff"
    text_color = "#000000"

    img = Image.new("RGB", (width, height), color=background_color)
    draw = ImageDraw.Draw(img)

    # Fonts – Pfade ggf. anpassen für andere Betriebssysteme
    font_path_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    font_path_regular = "/System/Library/Fonts/Supplemental/Arial.ttf"

    title_font = ImageFont.truetype(font_path_bold, 80)
    text_font = ImageFont.truetype(font_path_bold, 50)
    link_font = ImageFont.truetype(font_path_regular, 40)

    # Title: My Undo
    draw.text((60, 80), "My Undo", fill=text_color, font=title_font)

    # Formatierte Frage & Antwort
    margin = 60
    current_height = 220
    max_width = width - 2 * margin

    def draw_multiline(text, y_offset, font, spacing=10):
        lines = []
        words = text.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
            if w <= max_width:
                line = test_line
            else:
                lines.append(line)
                line = word
        lines.append(line)
        for l in lines:
            draw.text((margin, y_offset), l, font=font, fill=text_color)
            line_bbox = draw.textbbox((0, 0), l, font=font)
            line_height = line_bbox[3] - line_bbox[1]
            y_offset += line_height + spacing
        return y_offset

    current_height = draw_multiline(f"Q: {question}", current_height, text_font, spacing=15)
    current_height += 40  # Abstand zwischen Frage & Antwort
    current_height = draw_multiline(f"A: {answer}", current_height, text_font, spacing=15)

    # Link unten mittig
    link_text = "https://undoyourmind.com"
    link_bbox = draw.textbbox((0, 0), link_text, font=link_font)
    link_w = link_bbox[2] - link_bbox[0]
    link_h = link_bbox[3] - link_bbox[1]
    draw.text(
        ((width - link_w) / 2, height - link_h - 60),
        link_text,
        font=link_font,
        fill="#1a73e8"
    )

    # Speichern
    filename = f"static/share_{username}.png"
    img.save(filename)
    return "/" + filename