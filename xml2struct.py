import xmltodict

def xml2struct(file_path):
    with open(file_path, 'r') as file:
        xml_data = file.read()
        xml_dict = xmltodict.parse(xml_data)
    return xml_dict


