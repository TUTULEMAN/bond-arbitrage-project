from src.utils.common_imports import pd, os, ET
import requests

def fetch_ueld_curve_data():
    url= "https://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData"
    response = requests.get(url)
    if response.status_code!=200:
        raise Exception("Error loading and fetching information from FRED.")
    
    root = ET.fromstring(response.content)
    
    # Define namespaces to navigate the XML structure
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'd': 'http://schemas.microsoft.com/ado/2007/08/dataservices',
        'm': 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata'
    }
    
    data = []
    for entry in root.findall('atom:entr',ns):
        content = entry.find('atom:content',ns)
        properties=content.find('m:properties',ns)
        row={}
        for element in properties:
            tag = element.tag.split('}')[1]
            row[tag]=element.text
        data.append(row)

    df = pd.DataFrame(data)
    return df

def save_yield_curve_data(df,output_file):
    df.to_csv(output_file,index=False)
    print(f"Yield curve information has been saved to {output_file}")

def main():
    output_dir = os.path.join(os.path.dirname(__file__),'..','data','processed')#sending info to 'data/processed/'
    os.makedirs(output_dir,exist_ok=True)
    output_file = os.path.join(output_dir,'yield_curve_data.csv')

    #Fetching and saving
    df=fetch_ueld_curve_data()
    save_yield_curve_data(df,output_file)

if __name__ == '__main__':
    main()


