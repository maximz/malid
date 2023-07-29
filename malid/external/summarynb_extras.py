"""Extras for summarynb. Need to merge back into that library."""
from IPython.display import Image

# TODO: set .txt default handler to be plaintext()


def plaintext(fname):
    # return factory

    with open(fname, "r") as f:
        text = f.read()

        def template(*args, **kwargs):
            # convert to html
            return f"<pre>{text}</pre>"

        return template


def text_stdin(text):
    # TODO: rename
    # text fed through parameter, not loaded from file
    def template(*args, **kwargs):
        # convert to html
        return f"<pre>{text}</pre>"

    return template


def empty(width=None):
    # return factory
    def template(max_width, *args, **kwargs):
        # empty cell
        return f'<div style="min-width: {max_width if not width else width}px;"></div>'

    return template


def image_embed(fname):
    """Renders an image.
    :param img_src: Image filename.
    :type img_src: str
    :return: Template function that accepts a max pixel width integer and returns HTML.
    :rtype: function
    """

    def template(max_width, max_height):
        def convert_to_px_or_unset(optional_numeric_value):
            if not optional_numeric_value:
                return "inherit"
            return str(optional_numeric_value) + "px"

        max_width = convert_to_px_or_unset(max_width)
        max_height = convert_to_px_or_unset(max_height)

        image_data = Image(data=fname, embed=True)
        mimetype, base64_data = list(image_data._repr_mimebundle_()[0].items())[0]
        src_string = f"data:{mimetype};charset=utf-8;base64,{base64_data}"

        return """<img src="{img_src}" style="max-width: {max_width}; max-height: {max_height};" />""".format(
            img_src=src_string, max_width=max_width, max_height=max_height
        )

    return template
