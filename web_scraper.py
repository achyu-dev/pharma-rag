import json
import os
import requests
from bs4 import BeautifulSoup

DATASETS_PATH = "test"
DATASETS_MICROLABS_USA = os.path.join(DATASETS_PATH, "microlabs_usa")
os.makedirs(DATASETS_MICROLABS_USA, exist_ok=True) 

URLS = {
    "Acetazolamide Extended-Release Capsules": "https://www.microlabsusa.com/products/acetazolamide-extended-release-capsules/",
    "Amlodipine Besylate and Olmesartan Medoxomil Tablets": "https://www.microlabsusa.com/products/amlodipine-besylate-and-olmesartan-medoxomil-tablets/",
    "Amoxicillin and Clavulanate Potassium for Oral Suspension, USP": "https://www.microlabsusa.com/products/amoxicillin-and-clavulanate-potassium-for-oral-suspension-usp/",
    "Amoxicillin and Clavulanate Potassium Tablets, USP": "https://www.microlabsusa.com/products/amoxicillin-and-clavulanate-potassium-tablets-usp/",
    "Amoxicillin Capsules, USP": "https://www.microlabsusa.com/products/amoxicillin-capsules-usp/",
    "Aspirin and Extended-Release Dipyridamole Capsules": "https://www.microlabsusa.com/products/aspirin-and-extended-release-dipyridamole-capsules/",
    "Atorvastatin Calcium Tablets": "https://www.microlabsusa.com/products/atorvastatin-calcium-tablets/",
    "Bimatoprost Ophthalmic Solution": "https://www.microlabsusa.com/products/bimatoprost-ophthalmic-solution/",
    "Celecoxib capsules": "https://www.microlabsusa.com/products/celecoxib-capsules/",
    "Chlordiazepoxide Hydrochloride and Clidinium Bromide Capsules, USP": "https://www.microlabsusa.com/products/chlordiazepoxide-hydrochloride-and-clidinium-bromide-capsules-usp/",
    "Clindamycin Hydrochloride Capsules, USP": "https://www.microlabsusa.com/products/clindamycin-hydrochloride-capsules-usp/",
    "Dalfampridine Extended-Release Tablets": "https://www.microlabsusa.com/products/dalfampridine-extended-release-tablets/",
    "Diclofenac Sodium and Misoprostol Delayed-Release Tablets, USP": "https://www.microlabsusa.com/our-products/diclofenac-sodium-and-misoprostol-delayed-release-tablets-usp/",
    "Dorzolamide HCl and Timolol Maleate Ophthalmic Solution, USP": "https://www.microlabsusa.com/our-products/dorzolamide-hcl-and-timolol-maleate-ophthalmic-solution-usp/",
    "Dorzolamide HCl and Timolol Maleate Ophthalmic Solution, USP (Preservative Free)": "https://www.microlabsusa.com/products/dorzolamide-hcl-and-timolol-maleate-ophthalmic-solution-usppreservative-free/",
    "Dorzolamide HCl Ophthalmic Solution, USP": "https://www.microlabsusa.com/products/dorzolamide-hcl-ophthalmic-solution-usp/",
    "Famotidine for Oral Suspension, USP": "https://www.microlabsusa.com/products/famotidine-for-oral-suspension-usp/",
    "Fenofibric Acid Delayed-Release Capsules": "https://www.microlabsusa.com/products/fenofibric-acid-delayed-release-capsules/",
    "Formoterol Fumarate Inhalation Solution": "https://www.microlabsusa.com/products/acetazolamide-extended-release-capsules-2/",
    "Glimepiride Tablets, USP": "https://www.microlabsusa.com/products/glimepiride-tablets-usp/",
    "Ketorolac Tromethamine Ophthalmic Solution": "https://www.microlabsusa.com/products/ketorolac-tromethamine-ophthalmic-solution/",
    "Levocetirizine Dihydrochloride Tablets, USP": "https://www.microlabsusa.com/products/levocetirizine-dihydrochloride-tablets-usp/",
    "Mefenamic Acid Capsules, USP": "https://www.microlabsusa.com/products/mefenamic-acid-capsules-usp/",
    "Metformin Hydrochloride Extended-Release Tablets, USP": "https://www.microlabsusa.com/products/metformin-hydrochloride-extended-release-tablets-usp/",
    "Metformin Hydrochloride Oral Solution": "https://www.microlabsusa.com/products/metformin-hydrochloride-oral-solution/",
    "Methenamine Hippurate Tablets, USP": "https://www.microlabsusa.com/products/methenamine-hippurate-tablets-usp/",
    "Olmesartan Medoxomil Tablets, USP": "https://www.microlabsusa.com/products/olmesartan-medoxomil-tablets-usp/",
    "Piroxicam Capsules, USP": "https://www.microlabsusa.com/products/piroxicam-capsules-usp/",
    "Ramelteon Tablets": "https://www.microlabsusa.com/products/ramelteon-tablets/",
    "Ranolazine Extended-Release Tablets": "https://www.microlabsusa.com/products/ranolazine-extended-release-tablets/",
    "Rasagiline Tablets": "https://www.microlabsusa.com/products/rasagiline-tablets/",
    "Roflumilast Tablets": "https://www.microlabsusa.com/products/roflumilast/",
    "Rufinamide Tablets, USP": "https://www.microlabsusa.com/products/rufinamide-tablets-usp/",
    "Tafluprost Ophthalmic Solution": "https://www.microlabsusa.com/our-products/tafluprost-opthalmic-solution",
    "Telmisartan Tablets, USP": "https://www.microlabsusa.com/our-products/telmisartan-tablets-usp",
    "Timolol Maleate Ophthalmic Solution, USP": "https://www.microlabsusa.com/our-products/timolol-maleate-ophthalmic-solution-usp",
    "Timolol Maleate Ophthalmic Solution, USP (Preservative-Free)": "https://www.microlabsusa.com/our-products/timolol-maleate-ophthalmic-solution-usp-preservative-free",
    "Tobramycin Inhalation Solution, USP": "https://www.microlabsusa.com/our-products/tobramycin-inhalation-solution-usp",
    "Travoprost Ophthalmic Solution, USP": "https://www.microlabsusa.com/our-products/travoprost-ophthalmic-solution-usp"
}


def setup_prescribing_info_urls(urls_map):
    """
    Given a dict of product name (e.g. "Celecoxib capsules") and its corresponding URL, load the URL content,
    parse with BeautifulSoup to obtain the child URL for "Prescribing Information". Retrieve the child html content
    and create a soup object. Create a new dict with all these info and return these.
    :param urls_map: dict of product name (e.g. "Celecoxib capsules") and its corresponding URL
    :return: a dict updated_urls that maps product name to its url, child url and child soup where the child is
             the node containing "Prescribing Information"
    """
    updated_urls = {}

    for key, value in urls_map.items():
        # print("Processing: ", key)
        got = False
        updated_urls[key] = {
            "product_url": value,
        }
        data = requests.get(value)
        soup = BeautifulSoup(data.text, "html.parser")
        h2 = soup.findAll("h2")  # we know that "Prescribing Information" is enclosed by <h2> <a .../>

        for h2_item in h2:
            txt = h2_item.get_text()
            if txt is not None:
                if txt.strip().lower() == "Prescribing Information".lower():
                    child_url = h2_item.findAll("a")
                    if child_url:
                        href = child_url[0].get("href")
                        updated_urls[key]["prescribing_info_url"] = href
                        # print(href)
                        html = requests.get(href)
                        prescribing_soup = BeautifulSoup(html.text, "html.parser")
                        # print(prescribing_soup)
                        updated_urls[key]["prescribing_soup"] = prescribing_soup
                        got = True
            if got:  # we got the url and soup for "Prescribing Information" and so we break
                break

    return updated_urls


def find_elements_with_text(soup):
    # Find all elements that have text content
    elements_with_text = []
    for element in soup.find_all(True):  # True captures all tags
        if element.name not in ["script", "style"]:
            if element.string or element.get_text(strip=True):  # Check for non-empty text
                elements_with_text.append(element)

    # Print the elements and their text content
    for elem in elements_with_text:
        print(f"Tag: {elem.name}, Text: {elem.get_text(strip=True)}")

    return


def get_text_below_anchor_with_special_handling(a_tag):
    result_text = []

    # Loop through all siblings after the <a> tag
    for sibling in a_tag.find_next_siblings():
        if sibling.name == "div":
            childs = sibling.children
            for child in childs:
                if child.name is not None:
                    if child.name.lower() == "table":
                        # print("processing table...")
                        # Process table content
                        table_content = []
                        rows = sibling.find_all('tr')
                        for row in rows:
                            cells = [cell.get_text(strip=False) for cell in row.find_all(['td', 'th'])]
                            # Join cells with a single space separator
                            table_content.append(" ".join(cells))
                        result_text.append("\n".join(table_content))

                    elif child.name.lower() == "img":
                        # Process image content
                        img_src = sibling.get('src', 'No src attribute')
                        img_alt = sibling.get('alt', 'No alt text')
                        result_text.append(f"Image: [src={img_src}, alt={img_alt}]")
                    else:
                        result_text.append(sibling.get_text(strip=True))

    # Join and return the result
    return "\n".join(result_text)


def get_all_sections(soup):
    atags = soup.findAll("a")
    info = dict()

    for atag in atags:
        if atag:
            at = atag.get("id")

            if at and at.startswith("anch_dj_dj-dj"):
                txt = get_text_below_anchor_with_special_handling(atag)
                info[atag.get_text()] = txt
    return info


def process_prescribing_soup(name, soup):
    """
    This takes input as product name and its soup and returns the parsed content for "Prescribing Information"
    :param name: product name
    :param soup: bs4 soup object
    :return: parsed content as dict
    """
    # div = soup.find('div', class_='drug-label-sections')
    results = get_all_sections(soup)
    results["product_name"] = name
    return results


def create_dataset_file(pth, result):
    fname = os.path.join(pth, result["product_name"] + ".json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    return


if __name__ == '__main__':
    modified_urls = setup_prescribing_info_urls(URLS)
    for k, v in modified_urls.items():
        results = process_prescribing_soup(k, v["prescribing_soup"])
        create_dataset_file(DATASETS_MICROLABS_USA, results)

        # print("-" * 100)
        # print(results.keys())

    # print(results)
