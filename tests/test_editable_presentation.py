from xml.etree import ElementTree

from PIL import Image

from presentation.build_editable_sst_downscaling_deck import (
    Slide,
    _content_types,
    _fitted_box,
    _slide_xml,
    build_slides_expanded,
)


def test_slide_xml_keeps_text_and_diagram_objects_editable():
    slide = Slide("title", "section")
    slide.text(0.5, 0.5, 4.0, 0.8, "editable explanation", size=18)
    slide.box(1.0, 2.0, 2.0, 1.0, "editable block")
    slide.line(3.0, 2.5, 4.0, 2.5)
    root = ElementTree.fromstring(_slide_xml(slide, {}))
    namespace = {
        "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    }
    assert len(root.findall(".//p:sp", namespace)) == 2
    assert len(root.findall(".//p:cxnSp", namespace)) == 1
    assert len(root.findall(".//p:pic", namespace)) == 0
    assert [node.text for node in root.findall(".//a:t", namespace)] == [
        "editable explanation",
        "editable block",
    ]


def test_contained_picture_preserves_aspect_ratio(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (400, 200), "white").save(path)
    x, y, width, height = _fitted_box(path, 1.0, 1.0, 4.0, 4.0, contain=True)
    assert (x, y, width, height) == (1.0, 2.0, 4.0, 2.0)


def test_content_types_support_animated_gif():
    content = _content_types(1, {"gif", "png"})
    assert 'Extension="gif" ContentType="image/gif"' in content
    assert 'PartName="/ppt/slides/slide1.xml"' in content


def test_expanded_deck_has_requested_50_slide_narrative():
    slides = build_slides_expanded()
    assert len(slides) == 50
    assert slides[0].title == "AI SST downscaling for Australian boundary currents"
    assert slides[-2].title == "Conclusions"
    assert slides[-1].title == "What remains to be done"
    assert any(slide.title.startswith("Feature matching and adversarial") for slide in slides)
    assert any(slide.title.startswith("FiLM gives memory") for slide in slides)
    temporal = next(slide for slide in slides if slide.title.startswith("One-year comparison"))
    text = "\n".join(
        item.data.get("text", "") for item in temporal.items
        if item.kind in ("text", "box")
    )
    assert "GAN-v3 (historical + future)" in text
    assert "Residual-memory Flow-AR" in text
