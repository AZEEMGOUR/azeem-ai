from flask import Flask, request, jsonify
import os, json, faiss, numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from flask_cors import CORS  # 👈 import karo


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app) 
ENC = tiktoken.get_encoding("cl100k_base")



if __name__ == "__main__":
    app.run(debug=True)
