import re
import os

def analyze_repomix():
    xml_path = '/Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/repomix/repomix-output.xml'
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} does not exist.")
        return

    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all <file path="...">...</file> blocks.
    # Since XML parsing can sometimes fail on unescaped symbols in raw code,
    # we can use a regex to extract file paths and their contents safely.
    pattern = re.compile(r'<file path="([^"]+)">([\s\S]*?)</file>')
    matches = pattern.findall(content)

    if not matches:
        print("No files found in repomix-output.xml using regex pattern.")
        return

    files_info = []
    total_chars = 0

    for path, file_content in matches:
        char_count = len(file_content)
        total_chars += char_count
        files_info.append({
            'path': path,
            'chars': char_count,
            'bytes': len(file_content.encode('utf-8'))
        })

    # Sort files by char count descending
    files_info.sort(key=lambda x: x['chars'], reverse=True)

    print(f"Total Files Indexed: {len(files_info)}")
    print(f"Total Character Count: {total_chars:,} chars\n")

    # Group by top-level directory
    directory_groups = {}
    for info in files_info:
        # Normalize path
        normalized = info['path']
        if normalized.startswith('../'):
            normalized = normalized[3:]
        
        parts = normalized.split('/')
        if len(parts) > 1:
            group_name = parts[0] + '/'
        else:
            group_name = 'root/'
            
        if group_name not in directory_groups:
            directory_groups[group_name] = {'count': 0, 'chars': 0, 'bytes': 0}
        directory_groups[group_name]['count'] += 1
        directory_groups[group_name]['chars'] += info['chars']
        directory_groups[group_name]['bytes'] += info['bytes']

    print("=== BREAKDOWN BY DIRECTORY ===")
    print(f"{'Directory':<15} | {'Files':<6} | {'Char Count':<12} | {'Estimated Tokens':<16} | {'% of Total':<10}")
    print("-" * 69)
    for dir_name, stats in sorted(directory_groups.items(), key=lambda x: x[1]['chars'], reverse=True):
        percentage = (stats['chars'] / total_chars) * 100 if total_chars > 0 else 0
        est_tokens = stats['chars'] // 4  # Rough estimate (4 chars/token)
        print(f"{dir_name:<15} | {stats['count']:<6} | {stats['chars']:<12,} | {est_tokens:<16,} | {percentage:.2f}%")
    print("\n")

    print("=== TOP 25 LARGEST FILES ===")
    print(f"{'Rank':<4} | {'File Path':<70} | {'Char Count':<12} | {'% of Total':<10}")
    print("-" * 105)
    for idx, info in enumerate(files_info[:25]):
        percentage = (info['chars'] / total_chars) * 100 if total_chars > 0 else 0
        # clean path (remove ../ if present)
        clean_path = info['path']
        if clean_path.startswith('../'):
            clean_path = clean_path[3:]
        print(f"{idx+1:<4} | {clean_path:<70} | {info['chars']:<12,} | {percentage:.2f}%")

if __name__ == '__main__':
    analyze_repomix()
