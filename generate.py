import os
import re
import requests
from transformers import pipeline  # For text summarization
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

# Set device to CPU (since CUDA is unavailable)
device = -1  # -1 means CPU, 0 would be for GPU if available

# Load Stable Diffusion Model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)  # Use float32 on CPU
pipe.to("cpu")

# Load a smaller text summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Function to search for books in Project Gutenberg
def search_gutenberg(query):
    url = f"https://gutendex.com/books/?search={query}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data["count"] > 0:
            book = data["results"][0]  # Take the first result
            title = book["title"]
            authors = ", ".join([author["name"] for author in book["authors"]])
            
            # Look for a valid text format
            formats = book["formats"]
            text_url = formats.get("text/plain; charset=utf-8") or formats.get("text/plain") or formats.get("text/html")
            
            if text_url:
                return title, authors, text_url
    return None, None, None

# Function to download book text
def download_book(text_url):
    """Download the book content."""
    if text_url:
        response = requests.get(text_url)
        if response.status_code == 200:
            return response.text
    return None

# Helper function to clean and extract book text
def clean_book_text(book_text):
    """Remove front matter, back matter, and unnecessary whitespace."""
    start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK .+ \*\*\*", book_text)
    end = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK .+ \*\*\*", book_text)
    if start and end:
        book_text = book_text[start.end():end.start()]
    return book_text.strip()

# Function to split book text into chapters and extract meaningful paragraphs
def extract_passages(book_text, max_length=400, num_passages=3):
    """Extract logical and meaningful passages from the book."""
    # Clean the book text
    book_text = clean_book_text(book_text)

    # Split the text into chapters based on typical chapter indicators
    chapters = re.split(r"(?:Chapter|CHAPTER|CHAP\.|CHAPITRE|Capítulo) [IVXLCDM0-9]+", book_text)
    meaningful_passages = []

    # Process each chapter and extract paragraphs
    for chapter in chapters:
        paragraphs = chapter.split("\n\n")  # Split into paragraphs
        for paragraph in paragraphs:
            clean_paragraph = paragraph.strip()
            if len(clean_paragraph) > 100:  # Only keep paragraphs that are sufficiently long
                meaningful_passages.append(clean_paragraph)
                if len(meaningful_passages) >= num_passages:
                    return meaningful_passages[:num_passages]
    
    return meaningful_passages[:num_passages]

# Function to create a refined image prompt from a text passage
def create_image_prompt(text):
    """Generate a visual description prompt based on the text."""
    # Split the text into manageable chunks if it exceeds the token limit
    max_tokens = 1024  # BART's max token limit
    text_chunks = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
    
    # Summarize each chunk and combine the summaries
    summarized_text = ""
    for chunk in text_chunks:
        summary = summarizer(chunk, max_length=50, min_length=25, do_sample=False)[0]["summary_text"]
        summarized_text += summary + " "
    
    # Generate a single prompt from the summarized text
    return f"{summarized_text.strip()}, detailed, vibrant colors, fantasy art, cinematic lighting, highly detailed"

# Function to generate images based on a text prompt
def generate_image(prompt, output_path="generated_image.jpeg"):
    """Generate and save an image based on the prompt."""
    image = pipe(prompt, guidance_scale=7.5).images[0]
    image = image.resize((1024, 758))  # Resize for Kindle-friendly dimensions
    image.save(output_path, format="JPEG", quality=95)
    print(f"Image saved as {output_path}")

# Function to orchestrate the entire process of fetching a book, processing it, and generating images
def process_book(query):
    """Fetch book, process passages, and generate images."""
    # Step 1: Search for the book on Gutenberg
    title, authors, text_url = search_gutenberg(query)
    if not text_url:
        print(f"No book found for '{query}'.")
        return

    # Step 2: Download the book text
    print(f"Downloading '{title}' by {authors}")
    book_text = download_book(text_url)
    if not book_text:
        print("Failed to download book text.")
        return

    # Step 3: Extract meaningful passages
    passages = extract_passages(book_text)
    if not passages:
        print("No meaningful passages found.")
        return

    # Step 4: Generate images for each passage
    os.makedirs("book_images", exist_ok=True)
    for i, passage in enumerate(passages):
        prompt = create_image_prompt(passage)
        print(f"Generating image with prompt: {prompt}")
        
        safe_title = re.sub(r"[^a-zA-Z0-9]", "_", title)  # Make filename safe
        generate_image(prompt, f"book_images/{safe_title}_{i+1}.jpeg")

# Example: Run the process with "Pride and Prejudice"
process_book("Pride and Prejudice")
