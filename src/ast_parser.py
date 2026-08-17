import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from tree_sitter_languages import get_parser, get_language


# =====================================================================
# 1. Definisi Struktur Data Chunking
# =====================================================================
class CodeChunk(BaseModel):
    id: str
    file_path: str
    code_type: str  # 'function', 'class', 'method', 'interface', 'type_alias', 'file', 'file_block', dll.
    name: str
    content: str
    start_line: int
    end_line: int


# =====================================================================
# 2. Pemetaan ekstensi -> grammar tree-sitter
# =====================================================================
# Catatan: "typescript" dan "tsx" adalah grammar BERBEDA di tree-sitter.
# .tsx WAJIB pakai grammar "tsx" agar sintaks JSX-nya bisa di-parse.
EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
}

# Grammar mana yang berbagi query yang sama (TS dan TSX punya node type identik
# untuk interface/type-alias/class-field, JS tidak punya keduanya).
QUERY_FAMILY_MAP: Dict[str, str] = {
    "python": "python",
    "typescript": "typescript",
    "tsx": "typescript",
    "javascript": "javascript",
}

# Ambang batas baris untuk fallback chunking saat AST gagal / kosong.
FALLBACK_CHUNK_LINES = 200

# Label ramah-manusia untuk tiap suffix capture "*_body"
CODE_TYPE_LABELS = {
    "func": "function",
    "method": "method",
    "class": "class",
    "interface": "interface",
    "type": "type_alias",
}


# =====================================================================
# 3. Query Tree-sitter per keluarga bahasa
# =====================================================================
QUERIES: Dict[str, str] = {
    "python": """
        (function_definition name: (identifier) @func_name) @func_body
        (class_definition name: (identifier) @class_name) @class_body
    """,
    # TypeScript & TSX: function declaration, method, class, interface, type alias,
    # arrow function / function expression yang di-assign ke const/let, dan
    # arrow function sebagai class field (pola umum di komponen React/handler).
    "typescript": """
        (function_declaration name: (identifier) @func_name) @func_body
        (method_definition name: (property_identifier) @method_name) @method_body
        (class_declaration name: (type_identifier) @class_name) @class_body
        (interface_declaration name: (type_identifier) @interface_name) @interface_body
        (type_alias_declaration name: (type_identifier) @type_name) @type_body
        (variable_declarator
          name: (identifier) @func_name
          value: (arrow_function)) @func_body
        (variable_declarator
          name: (identifier) @func_name
          value: (function)) @func_body
        (public_field_definition
          name: (property_identifier) @method_name
          value: (arrow_function)) @method_body
    """,
    # JavaScript/JSX: sama seperti TS tapi tanpa interface/type-alias, dan nama
    # field class pakai node type "field_definition" (bukan "public_field_definition").
    "javascript": """
        (function_declaration name: (identifier) @func_name) @func_body
        (method_definition name: (property_identifier) @method_name) @method_body
        (class_declaration name: (identifier) @class_name) @class_body
        (variable_declarator
          name: (identifier) @func_name
          value: (arrow_function)) @func_body
        (variable_declarator
          name: (identifier) @func_name
          value: (function)) @func_body
        (field_definition
          property: (property_identifier) @method_name
          value: (arrow_function)) @method_body
    """,
}


class ASTCodeParser:
    """
    Parser AST multi-bahasa. Grammar dipilih otomatis berdasarkan ekstensi file
    (bukan satu bahasa hardcoded), dan setiap file yang gagal di-chunk lewat AST
    (bahasa tidak didukung, query tidak menemukan node, atau parsing error) akan
    diproses lewat fallback chunking supaya isinya tetap masuk index.
    """

    def __init__(self):
        self._parsers: Dict[str, Any] = {}
        self._languages: Dict[str, Any] = {}
        self._queries: Dict[str, Any] = {}

    def _get_parser_and_query(self, lang_name: str):
        if lang_name not in self._parsers:
            self._parsers[lang_name] = get_parser(lang_name)
            self._languages[lang_name] = get_language(lang_name)

            query_family = QUERY_FAMILY_MAP[lang_name]
            self._queries[lang_name] = self._languages[lang_name].query(QUERIES[query_family])

        return self._parsers[lang_name], self._queries[lang_name]

    @staticmethod
    def _is_inside_class(node) -> bool:
        """Dipakai untuk membedakan 'function' top-level vs 'method' di dalam class (khusus Python)."""
        current = node.parent
        while current is not None:
            if current.type == "class_definition":
                return True
            current = current.parent
        return False

    def _fallback_chunk_file(self, file_path: str, code_content: str) -> List[CodeChunk]:
        """
        Dipakai saat AST parsing gagal total atau tidak menghasilkan chunk apa pun
        (bahasa tidak dikenali, file kosong secara struktural, error parsing, dll).
        Tanpa fallback ini, seluruh isi file akan hilang begitu saja dari index.
        """
        lines = code_content.splitlines()
        if not lines:
            return []

        chunks: List[CodeChunk] = []
        file_name = os.path.basename(file_path)

        if len(lines) <= FALLBACK_CHUNK_LINES:
            chunks.append(
                CodeChunk(
                    id=f"{file_path}_file_{file_name}_1",
                    file_path=file_path,
                    code_type="file",
                    name=file_name,
                    content=code_content,
                    start_line=1,
                    end_line=len(lines),
                )
            )
            return chunks

        # File besar: pecah per blok baris supaya tetap dalam batas ukuran embedding
        for block_idx, start in enumerate(range(0, len(lines), FALLBACK_CHUNK_LINES), start=1):
            block_lines = lines[start:start + FALLBACK_CHUNK_LINES]
            start_line = start + 1
            end_line = start + len(block_lines)
            chunks.append(
                CodeChunk(
                    id=f"{file_path}_file_block_{file_name}_{block_idx}",
                    file_path=file_path,
                    code_type="file_block",
                    name=f"{file_name} (baris {start_line}-{end_line})",
                    content="\n".join(block_lines),
                    start_line=start_line,
                    end_line=end_line,
                )
            )
        return chunks

    def parse_file(self, file_path: str) -> List[CodeChunk]:
        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code_content = f.read()

        ext = os.path.splitext(file_path)[1]
        lang_name = EXTENSION_LANGUAGE_MAP.get(ext)

        if lang_name is None:
            # Ekstensi tidak dikenali grammar manapun -> langsung fallback.
            return self._fallback_chunk_file(file_path, code_content)

        chunks: List[CodeChunk] = []
        try:
            parser, query = self._get_parser_and_query(lang_name)
            tree = parser.parse(bytes(code_content, "utf8"))
            code_bytes = code_content.encode("utf8")

            processed_node_ids = set()

            for _pattern_idx, captures in query.matches(tree.root_node):
                body_node = None
                body_capture_name = None
                name_node = None

                for capture_name, node in captures.items():
                    if capture_name.endswith("_body"):
                        body_node = node
                        body_capture_name = capture_name
                    elif capture_name.endswith("_name"):
                        name_node = node

                if body_node is None:
                    continue
                if body_node.id in processed_node_ids:
                    continue
                processed_node_ids.add(body_node.id)

                suffix = body_capture_name.replace("_body", "")
                code_type = CODE_TYPE_LABELS.get(suffix, suffix)

                # Python tidak punya node "method_definition" sendiri, jadi function
                # yang bersarang di dalam class dilabeli ulang jadi "method".
                if lang_name == "python" and code_type == "function" and self._is_inside_class(body_node):
                    code_type = "method"

                name = (
                    code_bytes[name_node.start_byte:name_node.end_byte].decode("utf8")
                    if name_node is not None else "anonymous"
                )
                chunk_text = code_bytes[body_node.start_byte:body_node.end_byte].decode("utf8")

                chunks.append(
                    CodeChunk(
                        id=f"{file_path}_{code_type}_{name}_{body_node.start_point[0] + 1}",
                        file_path=file_path,
                        code_type=code_type,
                        name=name,
                        content=chunk_text,
                        start_line=body_node.start_point[0] + 1,
                        end_line=body_node.end_point[0] + 1,
                    )
                )
        except Exception as e:
            print(f"  [AST] Gagal memparsing '{file_path}' sebagai {lang_name}: {e}. Menggunakan fallback chunking.")
            return self._fallback_chunk_file(file_path, code_content)

        if not chunks:
            # Query tidak menemukan node apa pun (mis. file cuma berisi konstanta,
            # atau pola kode yang belum dicover query) -> tetap index isinya.
            return self._fallback_chunk_file(file_path, code_content)

        return chunks


# =====================================================================
# 4. Uji Coba Parser
# =====================================================================
if __name__ == "__main__":
    parser = ASTCodeParser()

    samples = {
        "sample.ts": """
class UserService {
    async getUser(id: string) {
        return { id, name: "Arvel" };
    }

    async deleteUser(id: string) {
        console.log("Deleting user", id);
    }
}

export const useUser = (id: string) => {
    return { id };
};

function calculateDiscount(price: number): number {
    return price * 0.1;
}
""",
        "sample.py": """
class UserService:
    def __init__(self, db):
        self.db = db

    async def get_user(self, user_id):
        return self.db.get(user_id)

def calculate_discount(price):
    return price * 0.1
""",
        "sample.tsx": """
export const Button = ({ label }: { label: string }) => {
    return <button>{label}</button>;
};
""",
    }

    for sample_file, content in samples.items():
        if not os.path.exists(sample_file):
            with open(sample_file, "w") as f:
                f.write(content)

        results = parser.parse_file(sample_file)
        print(f"--- {sample_file}: BERHASIL MEMOTONG {len(results)} CHUNK ---")
        for chunk in results:
            print(f"  Type: {chunk.code_type} | Name: {chunk.name} | Lines: {chunk.start_line}-{chunk.end_line}")