from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import  ChatOpenAI
import base64

class VisionLanguageAgentTools:
  """
  A class containing static methods for vision and language tasks.
  """

  @staticmethod
  @tool
  def clickNewImage() -> str:
    """Use camera to take picture of the current scenario and gives you the image path """
    return "demo.jpeg"

  @staticmethod
  def encoded_image(image_path):
    with open(image_path, "rb") as image_file:
      image_data = image_file.read()
      encoded_image = base64.b64encode(image_data).decode("utf-8")
      return encoded_image

  @staticmethod
  @tool
  def VisualQuestionAnswer(query: str, image_path : str) -> str:
    """ Takes a query which is the question users asks about the image and image_path which is the path of the image"""
    with open(image_path, "rb") as image_file:
      image_data = image_file.read()
      encoded_image = base64.b64encode(image_data).decode("utf-8")
    
    prompt = [
      HumanMessage(
        content=[
          {"type": "text", "text": query},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{encoded_image}"
            },
          },
        ],
      ),
    ]
    model = ChatOpenAI(model="gpt-4o")
    return model.invoke(prompt)
