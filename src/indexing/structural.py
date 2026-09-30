from pathlib import Path
from tree_sitter import Language, Parser
import tree_sitter_javascript as ts_javascript
from ..models import StructuralIndex, Symbol, CallEdge


class JavaScriptStructuralIndexer:

    def __init__(self):

        self.parser = Parser(
            Language(ts_javascript.language())
        )

    def index_file(
        self,
        file_path: str,
        content: str,
        index: StructuralIndex,
    ):

        tree = self.parser.parse(
            content.encode("utf-8")
        )

        self._walk(
            tree.root_node,
            content,
            file_path,
            index,
            current_function="",
        )

    def _walk(
        self,
        node,
        content,
        file_path,
        index,
        current_function,
    ):

        node_type = node.type

        if node_type in {
            "function_declaration",
            "method_definition",
            "arrow_function",
        }:

            name = self._get_function_name(node, content)

            if name:

                symbol = Symbol(
                    name=name,
                    kind="function",
                    file_path=file_path,
                    line=node.start_point[0] + 1,
                )

                index.add_symbol(symbol)

                current_function = name

        elif node_type == "class_declaration":

            name = self._get_identifier(node, content)

            if name:

                index.add_symbol(
                    Symbol(
                        name=name,
                        kind="class",
                        file_path=file_path,
                        line=node.start_point[0] + 1,
                    )
                )

            heritage = node.child_by_field_name("class")

            if heritage:
                pass

        elif node_type == "call_expression":

            function_node = node.child_by_field_name(
                "function"
            )

            if function_node:

                callee = self._node_text(
                    function_node,
                    content,
                )

                if callee and current_function:

                    index.add_call(
                        CallEdge(
                            caller=current_function,
                            callee=callee,
                            file_path=file_path,
                            line=node.start_point[0] + 1,
                        )
                    )

        for child in node.children:

            self._walk(
                child,
                content,
                file_path,
                index,
                current_function,
            )

    @staticmethod
    def _node_text(node, content):

        data = content.encode("utf-8")

        return data[
            node.start_byte:node.end_byte
        ].decode("utf-8", errors="ignore")

    def _get_identifier(self, node, content):

        for child in node.children:

            if child.type == "identifier":
                return self._node_text(child, content)

        return ""

    def _get_function_name(self, node, content):

        name_node = node.child_by_field_name("name")

        if name_node:

            return self._node_text(
                name_node,
                content,
            )

        parent = node.parent

        if parent and parent.type == "variable_declarator":

            name_node = parent.child_by_field_name("name")

            if name_node:
                return self._node_text(
                    name_node,
                    content,
                )

        return ""