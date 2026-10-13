"""Tests del splitter de documentos (sin embeddings ni LLM)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import Document
from src.splitter import Chunk, _find_break, split_documents


def make_doc(text: str, source: str = "notas.txt") -> Document:
    return Document(text=text, metadata={"source": source})


class SplitDocumentsTest(unittest.TestCase):
    def test_doc_corto_un_solo_chunk(self):
        doc = make_doc("Hola, esto cabe en un chunk")
        chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[0].source, "notas.txt")

    def test_doc_largo_varios_chunks_y_solapamiento(self):
        text = "\n\n".join(f"Parrafo {i} " + "p " * 150 for i in range(6))
        doc = make_doc(text)
        chunks = split_documents([doc], chunk_size=400, chunk_overlap=80)
        self.assertGreater(len(chunks), 1)
        # Todos los chunks respetan el tamaño máximo (aprox, por strip)
        for c in chunks:
            self.assertLessEqual(len(c.text), 400)
        # Hay solapamiento: el texto final del chunk i aparece en el chunk i+1
        tail = chunks[0].text[-60:]
        self.assertTrue(
            any(tail in c.text for c in chunks[1:]),
            "los chunks deben solaparse",
        )

    def test_indices_secuenciales(self):
        text = "\n\n".join("p " * 150 for _ in range(8))
        chunks = split_documents([make_doc(text)], chunk_size=400, chunk_overlap=80)
        idx = [c.metadata["chunk_index"] for c in chunks]
        self.assertEqual(idx, list(range(len(chunks))))

    def test_vacio_y_solo_espacios(self):
        chunks = split_documents([make_doc("")], chunk_size=100, chunk_overlap=20)
        self.assertEqual(chunks, [])
        chunks = split_documents([make_doc("    \n\n   ")], chunk_size=50, chunk_overlap=10)
        self.assertEqual(chunks, [])

    def test_parametros_invalidos(self):
        doc = make_doc("texto")
        with self.assertRaises(ValueError):
            split_documents([doc], chunk_size=0)
        with self.assertRaises(ValueError):
            split_documents([doc], chunk_size=100, chunk_overlap=100)

    def test_metadata_origen_preservada(self):
        doc = Document(text="a " * 300, metadata={"source": "docx.pdf", "page": 3})
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=10)
        self.assertTrue(all(c.metadata.get("source") == "docx.pdf" for c in chunks))
        self.assertTrue(all(c.metadata.get("page") == 3 for c in chunks))

    def test_find_break_prefiere_doble_salto(self):
        window = "abc\n\nxyz"
        self.assertEqual(_find_break(window), 5)  # corta después de "\n\n"


class ChunkSourceTest(unittest.TestCase):
    def test_source_default(self):
        self.assertEqual(Chunk(text="x").source, "desconocido")


if __name__ == "__main__":
    unittest.main()
