import json
import argparse

def format_json_file(input_file):
    # Read the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    def format_number_array(numbers, indent_level=1):
        indent = "    " * indent_level
        # Format numbers in a grid-like pattern, 20 numbers per line
        lines = []
        current_line = []
        max_per_line = 20
        
        for num in numbers:
            # Format each number to have consistent width with sign
            if num >= 0:
                formatted_num = " " + str(num).rjust(2)
            else:
                formatted_num = str(num).rjust(3)
            current_line.append(formatted_num)
            
            if len(current_line) >= max_per_line:
                lines.append(indent + " ".join(current_line))
                current_line = []
        
        # Add any remaining numbers
        if current_line:
            lines.append(indent + " ".join(current_line))
        
        return "[\n" + ",\n".join(lines) + "\n" + "    " * (indent_level - 1) + "]"

    def custom_format(obj, level=0):
        indent = "    " * level
        
        if isinstance(obj, dict):
            items = []
            for key in sorted(obj.keys()):
                value = obj[key]
                formatted_value = custom_format(value, level + 1)
                items.append('{0}    "{1}": {2}'.format(indent, key, formatted_value))
            return "{{\n{0}\n{1}}}".format(",\n".join(items), indent)
        
        elif isinstance(obj, list):
            if not obj:
                return "[]"
            elif all(isinstance(x, (int, float)) for x in obj):
                return format_number_array(obj, level + 1)
            else:
                items = ["{0}    {1}".format(indent, json.dumps(item)) for item in obj]
                return "[\n{0}\n{1}]".format(",\n".join(items), indent)
        
        return json.dumps(obj)
    
    # Write back with custom formatting
    formatted_json = custom_format(data)
    with open(input_file, 'w') as f:
        f.write(formatted_json)
    
    print(f"Successfully formatted {input_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format JSON file with proper indentation')
    parser.add_argument('json_file', type=str, help='Path to the JSON file to format')
    args = parser.parse_args()
    
    format_json_file(args.json_file) 