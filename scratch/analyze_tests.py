import re
import os

def analyze_tests():
    xml_path = '/Users/guillermocasasgonzalez/repos/porous_NS_with_Gridap/repomix/repomix-output.xml'
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} does not exist.")
        return

    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to extract file paths and contents
    pattern = re.compile(r'<file path="([^"]+)">([\s\S]*?)</file>')
    matches = pattern.findall(content)

    test_files = []
    total_test_chars = 0

    for path, file_content in matches:
        # Normalize path
        clean_path = path
        if clean_path.startswith('../'):
            clean_path = clean_path[3:]
            
        if clean_path.startswith('test/'):
            char_count = len(file_content)
            total_test_chars += char_count
            test_files.append({
                'path': clean_path,
                'chars': char_count
            })

    # Sort test files by size descending
    test_files.sort(key=lambda x: x['chars'], reverse=True)

    print(f"Total Test Files: {len(test_files)}")
    print(f"Total Test Character Count: {total_test_chars:,} chars\n")

    # Group by tier/subfolder in test/
    groups = {}
    for item in test_files:
        # e.g., test/blitz/... or test/quick/... or test/extended/...
        parts = item['path'].split('/')
        if len(parts) > 2:
            group_name = 'test/' + parts[1] + '/'
        else:
            group_name = 'test/root/'
            
        if group_name not in groups:
            groups[group_name] = {'count': 0, 'chars': 0}
        groups[group_name]['count'] += 1
        groups[group_name]['chars'] += item['chars']

    print("=== BREAKDOWN BY TEST CATEGORY ===")
    print(f"{'Category':<18} | {'Files':<6} | {'Char Count':<12} | {'Estimated Tokens':<16} | {'% of Test Chars':<15}")
    print("-" * 75)
    for cat, stats in sorted(groups.items(), key=lambda x: x[1]['chars'], reverse=True):
        pct = (stats['chars'] / total_test_chars) * 100 if total_test_chars > 0 else 0
        est_tokens = stats['chars'] // 4
        print(f"{cat:<18} | {stats['count']:<6} | {stats['chars']:<12,} | {est_tokens:<16,} | {pct:.2f}%")
    print("\n")

    print("=== SORTED TEST FILES BY SIZE ===")
    print(f"{'Rank':<4} | {'Test File Path':<75} | {'Char Count':<12} | {'% of Test':<10}")
    print("-" * 110)
    for idx, item in enumerate(test_files):
        pct = (item['chars'] / total_test_chars) * 100 if total_test_chars > 0 else 0
        print(f"{idx+1:<4} | {item['path']:<75} | {item['chars']:<12,} | {pct:.2f}%")

if __name__ == '__main__':
    analyze_tests()
