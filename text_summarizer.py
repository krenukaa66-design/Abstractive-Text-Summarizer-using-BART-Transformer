from transformers import pipeline

def summarize_text(text, max_len=130, min_len=30):
    # Explicitly use BART (better quality)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    input_text = """
    Artificial Intelligence is the simulation of human intelligence processes by machines, 
    especially computer systems. These processes include learning (the acquisition of information 
    and rules for using the information), reasoning (using rules to reach approximate or definite 
    conclusions), and self-correction. Particular applications of AI include expert systems, 
    speech recognition, and machine vision.
    """

    print("Original Text:\n", input_text.strip())
    print("\n--- Summary ---\n")
    print(summarize_text(input_text))
