import re
import os



class MetaDataProvider:

    @staticmethod
    def get_metadata(file_path):
        """
        Parse a Vue or JS file to extract function names and class names.
        """
        

        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        result = f"file_name: {file_name} absolute_path: {file_path} file_extension: {file_extension} "
        if file_extension not in ['.vue', '.js']:
            return result
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if file_extension == '.vue':
            script_match = re.search(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
            if script_match:
                script_content = script_match.group(1)
            else:
                script_content = ""
        else:
            script_content = content
        
        class_names = []
        
        # Match traditional JS classes: "class ClassName {"
        class_matches = re.finditer(r'class\s+([A-Za-z_$][\w$]*)', script_content)
        for match in class_matches:
            class_names.append(match.group(1))
        
        # Match Vue component definitions: "export default { name: 'ComponentName' }"
        vue_component_match = re.search(r"name:\s*['\"]([^'\"]+)['\"]", script_content)
        if vue_component_match:
            class_names.append(vue_component_match.group(1))
        
        function_names = []
        
        # Traditional function declarations: "function funcName() {" 
        function_matches = re.finditer(r'function\s+([A-Za-z_$][\w$]*)\s*\(', script_content)
        for match in function_matches:
            function_names.append(match.group(1))
        
        # Arrow functions with variable assignment: "const funcName = () => {"
        arrow_func_matches = re.finditer(r'(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*\([^)]*\)\s*=>', script_content)
        for match in arrow_func_matches:
            function_names.append(match.group(1))
        
        # Object methods: "methodName() {"
        method_matches = re.finditer(r'(?<!\.|\"|\')([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*{', script_content)
        for match in method_matches:
            function_names.append(match.group(1))
        
        # Vue methods: "methods: { methodName() {"
        vue_methods_section = re.search(r'methods\s*:\s*{(.*?)}', script_content, re.DOTALL)
        if vue_methods_section:
            methods_content = vue_methods_section.group(1)
            vue_method_matches = re.finditer(r'([A-Za-z_$][\w$]*)\s*\([^)]*\)', methods_content)
            for match in vue_method_matches:
                function_names.append(match.group(1))
        
        reserved_words = {'if', 'else', 'for', 'while', 'switch', 'case', 'default', 'return'}
        function_names = [name for name in list(dict.fromkeys(function_names)) if name not in reserved_words]
        
        return result + f"function_names: {", ".join(function_names)} class_names: {" ".join(class_names)}"