from pyzotero import zotero
import bibtexparser
from bibtexparser.bwriter import BibTexWriter

# Configuration
API_KEY = None
GROUP_ID = '5219733'
OUTPUT_FILE = 'source/bibliography.bib'


def ensure_unique_keys(entries):
    """Ensure BibTeX keys are unique."""
    seen_keys = set()
    for entry in entries:
        original_key = entry.get('ID', 'no_id')
        key = original_key
        suffix = 1
        while key in seen_keys:
            key = f"{original_key}_{suffix}"
            suffix += 1
        entry['ID'] = key
        seen_keys.add(key)
    return entries

def export_zotero_to_bibtex(group_id, api_key=None, output_file='bibliography.bib'):
    """
    Exports the Zotero library to a BibTeX file.

    Parameters:
    - group_id: Zotero Group ID
    - api_key: Zotero API Key
    - output_file: The name of the output file (default is 'zotero_export.bib')
    """

    # Create a Zotero client instance
    zot = zotero.Zotero(group_id, 'group', api_key)

    # Fetch bibtex entries from Zotero
    bibtex_db = zot.everything(zot.items(format='bibtex'))
    ensure_unique_keys(bibtex_db.entries)
    
    # Convert bibtex entries to a string
    db = bibtexparser.bibdatabase.BibDatabase()
    db.entries = bibtex_db.entries
    writer = BibTexWriter()
    bibtex_str = bibtexparser.dumps(db, writer=writer)

    # Write the string to the specified output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(bibtex_str)
    
 
if __name__ == '__main__':
    export_zotero_to_bibtex(GROUP_ID, API_KEY, OUTPUT_FILE)
    print(F"Bibliography generation completed! Check file: {OUTPUT_FILE}")

