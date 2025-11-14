from bot_graph import workflow
from PIL import Image
from io import BytesIO


def show_workflow():

    png_data = workflow.get_graph().draw_mermaid_png()

    try:
        img = Image.open(BytesIO(png_data))
        img.show()  # Opens in default image viewer
    except Exception as e:
        print(f"Could not display image: {e}")


if __name__ == "__main__":
    show_workflow()