import xmltodict
import matlab.engine
# from matlab.engine import matlab.engine.setup()


def test_xml_to_dict_equivalence():
    # Prepare a sample XML file
    xml_content = '''
    <XMLname attrib1="Some value">
        <Element>Some text</Element>
        <DifferentElement attrib2="2">Some more text</DifferentElement>
        <DifferentElement attrib3="2" attrib4="1">Even more text</DifferentElement>
    </XMLname>
    '''
    file_path = 'sample.xml'
    with open(file_path, 'w') as file:
        file.write(xml_content)

    # Convert XML to dictionary using xml_to_dict function
    dict1 = xml_to_dict(file_path)

    # Convert XML to dictionary using xml2struct2 function (implemented in MATLAB)
    # For comparison purposes, you may need to write an equivalent Python function
    # or convert the MATLAB function to Python
    dict2 = convert_xml_to_dict_using_matlab_function(file_path)

    # Compare the resulting dictionary structures
    assert dict1 == dict2, "Dictionary structures from both functions are not equal"

    print("Unit test passed: Both functions produce equivalent dictionary structures")

def convert_xml_to_dict_using_matlab_function(file_path):
    # Start the MATLAB Engine session
    eng = matlab.engine.start_matlab()
    eng.addpath(r"\\10.99.68.52\Kiemendata\Valentina Matos\Dashboard Project\CODA Hub\base")

    # Call the MATLAB function with the provided file path
    dict2 = eng.xml2struct2(file_path)

    # Close the MATLAB Engine session
    eng.quit()

    return dict2

#same funciton in python
def xml_to_dict(file_path):
    with open(file_path, 'r') as file:
        xml_data = file.read()
        xml_dict = xmltodict.parse(xml_data)
    return xml_dict


# Run the unit test
test_xml_to_dict_equivalence()
