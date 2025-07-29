import fitz  # PyMuPDF
import csv
import re
import argparse
import os

# EXAMPLE USAGE: python3 flx_converter.py FLX_result.pdf -o results_flx.csv

DATE_REGEX = re.compile(r"\b(0[1-9]|1[0-2])/([0-2][0-9]|3[01])/([0-9]{2})\b")

def extract_rows_by_date_anchor(pdf_path, output_path):
    rows = []
    game_list = ['LOTTO', 'LOTTO DP', 'X2', 'X3', 'X4', 'X5']
    elimination_list = ['LOTTO DP', 'X2', 'X3', 'X4', 'X5']
    
    with fitz.open(pdf_path) as doc:
        for page in doc:
            dates = []
            values = []
            games = []
            date_index = 0
            MAX_DATE = '10/10/20'
            max_date_reached = False
            
            blocks = page.get_text("dict")["blocks"]

            # Flatten all spans with position
            elements = []
            for block in blocks:
                #print("NEW BLOCK")
                for line in block.get("lines", []):
                    #print("NEW LINE")
                    for span in line.get("spans", []):
                        #print("NEW SPAN")
                        text = span["text"].strip()
                        if text:
                            elements.append({
                                "text": text,
                                "x": span["bbox"][0],
                                "y": span["bbox"][1]
                            })

                            #print(f"text: {text}")

                            if DATE_REGEX.fullmatch(text):
                                #print(f"INSERT DATE: {text} (length: {len(dates)+1})")
                                dates.append(text)

                                #TODO check against date 10/10/2020, stop processing at this page
                                if text == MAX_DATE:
                                    max_date_reached = True
                                    
            # Flatten all spans with position
            elements = []
            for block in blocks:
                #print("NEW BLOCK")
                for line in block.get("lines", []):
                    #print("NEW LINE")
                    for span in line.get("spans", []):
                        #print("NEW SPAN")
                        text = span["text"].strip()
                        if text:
                            elements.append({
                                "text": text,
                                "x": span["bbox"][0],
                                "y": span["bbox"][1]
                            })

                            #print(f"text: {text}")

                            if text.isdigit():
                                # if we've already filled the array, now we should start appending by index
                                if len(dates) == len(values):   
                                    #print(f"CONCATE DIGIT: {values[date_index]} + {text} (length: {len(values)})")
                                    values[date_index] += " " + text
                                    date_index += 1 # update ahead of next usage

                                    if date_index == len(values):
                                        date_index = 0
                                else:
                                    #print(f"INSERT DIGIT: {text} (length: {len(values)+1})")
                                    values.append(text)
                            elif text in game_list:
                                #print(f"INSERT GAME: {text} (length: {len(games)+1})")
                                games.append(text)
                                                                        
            #print(f"dates = {dates}")
            #print(f"values = {values}")
            if len(dates) != len(values):
                print(f"FAIL! NUMBER OF DATES AND VALUES ARE NOT ALIGNED!")
                print(f"dates = {dates}")
                print(f"values = {values}")
                exit(1)
            for value in values:
                if len(value.split(" ")) != 6:
                    print(f"FAIL! EXPECTED ALL READ VALUES TO CONTAIN 6 NUMBERS")
                    exit(1)

            for index, date in enumerate(dates):
                if games[index] not in elimination_list:
                    rows.append(date + "," + values[index])
                #else:
                #    print(f"ELIMINATING DATA: {index}. {date} :: {values[index]} :: {games[index]}")

            print(f"rows count = {len(rows)}")
        
            if max_date_reached: break

    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        file.write('Draw Date,Winning Numbers')
        for line in rows:
            file.write('\n' + line)

    print(f"Extracted {len(rows)} rows to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract misaligned records from PDF")
    parser.add_argument("pdf", nargs="?", default="input.pdf", help="PDF file to extract from")
    parser.add_argument("-o", "--output", default="output.csv", help="CSV output file name")

    args = parser.parse_args()
    input_path = os.path.abspath(args.pdf)

    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        exit(1)

    # Route output next to input if output is just a filename
    output_path = args.output
    if not os.path.isabs(output_path) and os.path.dirname(output_path) == "":
        output_path = os.path.join(os.path.dirname(input_path), output_path)

    extract_rows_by_date_anchor(input_path, output_path)
