import os
from typing import List, Dict, Any
from pydantic import BaseModel
from tree_sitter_languages import get_parser, get_language

# 1. Definisi Struktur Data Chunking
class CodeChunk(BaseModel):
    id: str
    file_path: str
    code_type: str  # 'function', 'class', 'method', dll.
    name: str
    content: str
    start_line: int
    end_line: int

# 2. Fungsi Parser berbasis Tree-sitter
class ASTCodeParser:
    def __init__(self, language_name: str = "typescript"):
        self.language_name = language_name
        self.parser = get_parser(language_name)
        self.language = get_language(language_name)

    def parse_file(self, file_path: str) -> List[CodeChunk]:
        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            code_content = f.read()

        tree = self.parser.parse(bytes(code_content, "utf8"))
        root_node = tree.root_node
        chunks: List[CodeChunk] = []

        # Query Tree-sitter untuk menemukan fungsi dan kelas (Sesuaikan dengan sintaks bahasa)
        # Contoh query untuk TypeScript / JavaScript:
        query_scm = """
        (function_declaration name: (identifier) @func_name) @func_body
        (method_definition name: (property_identifier) @method_name) @method_body
        (class_declaration name: (type_identifier) @class_name) @class_body
        """
        
        try:
            query = self.language.query(query_scm)
            captures = query.captures(root_node)

            processed_nodes = set()

            for node, capture_name in captures:
                if capture_name in ["func_body", "method_body", "class_body"]:
                    if node.id in processed_nodes:
                        continue
                    processed_nodes.add(node.id)

                    # Ekstraksi informasi node
                    code_bytes = code_content.encode("utf8")
                    chunk_text = code_bytes[node.start_byte:node.end_byte].decode("utf8")
                    
                    code_type = capture_name.replace("_body", "")
                    
                    # Mencari nama fungsi/kelas dari child node
                    name = "anonymous"
                    for child in node.children:
                        if "identifier" in child.type:
                            name = code_bytes[child.start_byte:child.end_byte].decode("utf8")
                            break

                    chunks.append(
                        CodeChunk(
                            id=f"{file_path}_{code_type}_{name}_{node.start_point[0] + 1}",
                            file_path=file_path,
                            code_type=code_type,
                            name=name,
                            content=chunk_text,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                        )
                    )
        except Exception as e:
            print(f"Error parsing query: {e}")

        return chunks

# 3. Uji Coba Parser
if __name__ == "__main__":
    parser = ASTCodeParser("typescript")
    
    # Ganti dengan path file di repositori lokalmu
    sample_file = "sample.ts" 
    
    # Buat dummy file jika belum ada
    if not os.path.exists(sample_file):
        with open(sample_file, "w") as f:
            f.write("""
class UserService {
    async getUser(id: string) {
        return { id, name: "Arvel" };
    }

    async deleteUser(id: string) {
        console.log("Deleting user", id);
    }
}

function calculateDiscount(price: number): number {
    return price * 0.1;
}
            """)

    results = parser.parse_file(sample_file)
    
    print(f"--- BERHASIL MEMOTONG {len(results)} CHUNK AST ---")
    for chunk in results:
        print(f"\nType: {chunk.code_type} | Name: {chunk.name} | Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Content Snippet:\n{chunk.content[:80]}...")