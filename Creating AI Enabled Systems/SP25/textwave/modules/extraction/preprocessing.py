import os
import re
from glob import glob
import nltk


class DocumentProcessing:
    """
    A class used for processing documents including reading, trimming whitespace,
    and splitting documents into sentence chunks.

    Attributes:
        None
    
    Methods:
        __read_text_file(file_path: str) -> str:
            Reads the content of a text file.

        trim_white_space(text: str) -> str:
            Trims extra whitespace from the given text.

        sentence_chunking(document_filename: str, num_sentences: int, overlap_size: int = 0) -> list:
            Splits the document into chunks of specified number of sentences.

        fixed_length_chunking(document_filename: str, chunk_size: int, overlap_size: int = 2) -> list:
            Divides the document into fixed-size chunks of characters.
    """

    def __init__(self):
        """
        Initializes the DocumentProcessing class. No attributes are initialized.
        """
        pass

    def __read_text_file(self, file_path: str) -> str:
        """
        Reads the content of a text file.

        Args:
            file_path (str): The path to the text file.

        Returns:
            str: The content of the text file or an error message if an issue occurs.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            return f"The file at {file_path} was not found."
        except Exception as e:
            return f"An error occurred: {e}"

    def trim_white_space(self, text: str) -> str:
        """
        Trims extra whitespace from the given text.

        Args:
            text (str): The text to be trimmed.

        Returns:
            str: The trimmed text with unnecessary whitespaces removed.
        """
        return ' '.join(text.split())

    def sentence_chunking(self, document_filename: str, num_sentences: int, overlap_size: int = 0) -> list:
        """
        Splits the document into chunks of specified number of sentences.

        Args:
            document_filename (str): The filename of the document to be split.
            num_sentences (int): The number of sentences per chunk.
            overlap_size (int): Number of overlapping sentences between chunks.

        Returns:
            list: A list of sentence chunks.
        """
        text = self.__read_text_file(document_filename)

        if isinstance(text, str):
            text = self.trim_white_space(text)
            sentences = nltk.sent_tokenize(text)

            chunks = []
            i = 0
            while i < len(sentences):
                chunk = ' '.join(sentences[i:i + num_sentences])
                chunks.append(chunk)
                i += (num_sentences - overlap_size)

            return chunks
        return [text]

    def fixed_length_chunking(self, document_filename: str, chunk_size: int, overlap_size: int = 2) -> list:
        """
        Divides the document into fixed-size chunks of characters.

        Args:
            document_filename (str): The filename of the document to be split.
            chunk_size (int): Number of characters per chunk.
            overlap_size (int): Number of overlapping characters between chunks.

        Returns:
            list: A list of text chunks.
        """

        #########################################
        # TODO: Implement fixed_length_chunking()
        #########################################
    def fixed_length_chunking(self, document_filename: str, chunk_size: int, overlap_size: int = 2) -> list:
        text = self.__read_text_file(document_filename)
        if not isinstance(text, str):
            return [text]
        text = self.trim_white_space(text)
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i:i+chunk_size]
            chunks.append(chunk)
            i += (chunk_size - overlap_size)
        return chunks

    # ────────────────────────────────────────────────────────────────
    # NEW:  chunk pre‑loaded text instead of reading from disk
    # ────────────────────────────────────────────────────────────────
    def sentence_chunking_text(self, text: str,
                               num_sentences: int,
                               overlap_size: int = 0) -> list[str]:
        """
        Split *text* (already in memory) into chunks of `num_sentences`.
        """
        text = self.trim_white_space(text)
        sents = nltk.sent_tokenize(text)
        out, i = [], 0
        while i < len(sents):
            out.append(" ".join(sents[i:i + num_sentences]))
            i += max(1, num_sentences - overlap_size)
        return out

    def fixed_length_chunking_text(self, text: str,
                                   chunk_size: int,
                                   overlap_size: int = 2) -> list[str]:
        """
        Split *text* (already in memory) into fixed‑length character chunks.
        """
        text = self.trim_white_space(text)
        out, i = [], 0
        while i < len(text):
            out.append(text[i:i + chunk_size])
            i += max(1, chunk_size - overlap_size)
        return out


if __name__ == "__main__":
    processing = DocumentProcessing()

    # Example to split documents into sentence chunks
    chunks = processing.sentence_chunking("storage/S08_set3_a1.txt.clean", num_sentences=5, overlap_size=3)
    for idx, chunk in enumerate(chunks):
        print(idx, chunk)