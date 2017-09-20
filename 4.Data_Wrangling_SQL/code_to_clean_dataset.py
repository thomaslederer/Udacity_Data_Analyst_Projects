
# coding: utf-8

# In[1]:

import csv
import codecs
import pprint
import re
import xml.etree.cElementTree as ET

import cerberus

import schema

OSM_PATH = "testy.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema.schema

NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']


def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    waytags = []
    nodetags = []
    
    
    if element.tag == 'node':
        #This loop parses data for export to nodes.csv file.
        for attr in element.attrib:
            if attr in node_attr_fields:
                node_attribs[attr] = element.attrib[attr]
        
        checking_for_city_and_zip = []
        # list created to scan keys in element for zipcode and city
        
        citydict = {}            
        #Dictionary created for element's with zipcode but missing city data
        
        for tag in element.iter('tag'):
            checking_for_city_and_zip.append(tag.attrib['k'])
            #Scan all keys...
            
        if "addr:postcode" in checking_for_city_and_zip:
            #If zipcode in keys, create a city based upon that zipcode.
            for tag in element.iter('tag'):
                if tag.attrib['k'] == "addr:postcode":
                    zip = correctzip(tag.attrib['v'])
                    if zip != "11385":
                        city = zipcodemapping[zip][0] 
                    
                        if "addr:city" not in checking_for_city_and_zip:
                        #If city data not included, created here:

                            citydict['id'] = element.attrib['id']
                            citydict['key'] = "city"
                            citydict['value'] = city
                            citydict["type"] = "addr"
          
                    if zip == "11385":
                        #For ambigious zipcode, use city given in data.
                        for tag in element.iter('tag'):
                            if tag.attrib["k"] == "addr:city":
                                city = tag.attrib["v"]           
        
        if citydict:
            nodetags.append(citydict)
            #Add newly created city data to dictionary
                
        if "addr:postcode" not in checking_for_city_and_zip and "addr:city" in checking_for_city_and_zip:
            # If no zipcode, can at least scan city to make sure it's grammatically consistent
            for tag in element.iter('tag'):
                if tag.attrib['k'] == "addr:city":
                    city = correctcity(tag.attrib["v"])
        
         
            
        for tag in element.iter('tag'):            
        # Loop though tags again, this time adding to dictionary 
            
            
            unitdict = {}
            #Dictionary created for suites/unit/room numbers
            
            tagdict = {}
            # Dictionary for all other tags
            
            
         
            key = tag.attrib['k']
            m = problem_chars.search(key)
            if not m:
             
 
                # Id and value for other tags added to tagdict.
                tagdict['id'] = element.attrib['id']
                tagdict['value'] = tag.attrib['v']
                
                # Key and type for other tags added to tagdict.
                if key.find(":") == -1:
                    tagdict['key'] = key
                    tagdict['type'] = default_tag_type
                else:
                    pos = key.find(":")
                    tagdict['key'] = key[pos+1:]
                    tagdict['type'] = key[:pos]
                
                #This corrects the street type to non-abbreviated form and seperates out suite
                #if this value is included within the "street" key
                if tagdict['key'] == "street":
                    streetname = tagdict["value"]

                    street, suite = correctstreet(streetname)
                    tagdict["value"] = street
                    
                    #If "suite" value returned, new entry prepared.
                    if suite != "NA":
                        unitdict['id'] = element.attrib['id']
                        unitdict['key'] = "unit"
                        unitdict["value"] = suite
                        unitdict['type'] = "addr"

                if "key" in unitdict:
                # If unit dict not empty, append data to dictionary    
                    nodetags.append (unitdict)       
                        
                        
                #Correct housenumber to "address note" if not a number.
                if tagdict ["key"] == "housenumber":
                    if not is_number (tagdict['value']):
                        tagdict ["key"] = "address note"
                        
                # Corrected zipcode defined above added.
                if tagdict ["key"] == "postcode":
                    tagdict['value'] = zip 
                    
                    
                #If a value for city was included in the osm file, value corrected above added
                # to dictionary.  
                if tagdict["key"] == "city":
                    tagdict ["value"] = city
            
                
                # Funeral directors corrected as "amenity" key.
                if tagdict["key"] == "shop" and tagdict["value"] == "funeral_directors":
                    tagdict["key"] == "amenity"

                # Ice_cream and laundry corrected as "shop" key.
                if tagdict["key"] == "amenity" and (tagdict["value"] == "ice_cream" or tagdict["value"] == "laundry"):
                    tagdict["key"] = "shop"

                #Add all remaining tags to final dictionary 
                nodetags.append(tagdict)
        
                
        return {'node': node_attribs, 'node_tags': nodetags}
        
    #Repeat procedures to create dictionaries/csv files for way tags.
    if element.tag == 'way':
        for attr in element.attrib:
            if attr in way_attr_fields:
                way_attribs[attr] = element.attrib[attr]   
        
        counter = 0
        for tag in element.iter("nd"):
            nodedict = {}
            
                
            nodedict['id'] = element.attrib['id']
            nodedict['node_id'] = tag.attrib['ref']
            nodedict['position'] = counter
            
            way_nodes.append(nodedict)
            
            counter += 1
            
        
        checking_for_city_and_zip = []
        # list created to scan keys in element for zipcode and city
        
        
        citydict = {}            
        #Dictionary created for element's with zipcode but missing city data
        
        for tag in element.iter('tag'):
            checking_for_city_and_zip.append(tag.attrib['k'])
            #Scan all keys...
            
        if "addr:postcode" in checking_for_city_and_zip:
            #If zipcode in keys, create a city based upon that zipcode.
            for tag in element.iter('tag'):
                if tag.attrib['k'] == "addr:postcode":
                    zip = correctzip(tag.attrib['v'])
                    if zip != "11385":
                        city = zipcodemapping[zip][0] 
                    
                        if "addr:city" not in checking_for_city_and_zip:
                        #If city data not included, created here:

                            citydict['id'] = element.attrib['id']
                            citydict['key'] = "city"
                            citydict['value'] = city
                            citydict["type"] = "addr"
          
                    if zip == "11385":
                        #For ambigious zipcode, use city given in data.
                        for tag in element.iter('tag'):
                            if tag.attrib["k"] == "addr:city":
                                city = tag.attrib["v"]           
        
        if citydict:
            waytags.append(citydict)
            #Add newly created city data to dictionary
                
        if "addr:postcode" not in checking_for_city_and_zip and "addr:city" in checking_for_city_and_zip:
            # If no zipcode, can at least scan city to make sure it's grammatically consistent
            for tag in element.iter('tag'):
                if tag.attrib['k'] == "addr:city":
                    city = correctcity(tag.attrib["v"])
        
         
            
        for tag in element.iter('tag'):            
        # Loop though tags again, this time adding to dictionary 
            
            
            unitdict = {}
            #Dictionary created for suites/unit/room numbers
            
            tagdict = {}
            # Dictionary for all other tags
            
            
         
            key = tag.attrib['k']
            m = problem_chars.search(key)
            if not m:
             
 
                # Id and value for other tags added to tagdict.
                tagdict['id'] = element.attrib['id']
                tagdict['value'] = tag.attrib['v']
                
                # Key and type for other tags added to tagdict.
                if key.find(":") == -1:
                    tagdict['key'] = key
                    tagdict['type'] = default_tag_type
                else:
                    pos = key.find(":")
                    tagdict['key'] = key[pos+1:]
                    tagdict['type'] = key[:pos]
                
                #This corrects the street type to non-abbreviated form and seperates out suite
                #if this value is included within the "street" key
                if tagdict['key'] == "street":
                    streetname = tagdict["value"]

                    street, suite = correctstreet(streetname)
                    tagdict["value"] = street
                    
                    #If "suite" value returned, new entry prepared.
                    if suite != "NA":
                        unitdict['id'] = element.attrib['id']
                        unitdict['key'] = "unit"
                        unitdict["value"] = suite
                        unitdict['type'] = "addr"

                if "key" in unitdict:
                # If unit dict not empty, append data to dictionary    
                    waytags.append (unitdict)       
                        
                        
                #Correct housenumber to "address note" if not a number.
                if tagdict ["key"] == "housenumber":
                    if not is_number (tagdict['value']):
                        tagdict ["key"] = "address note"
                        
                # Corrected zipcode defined above added.
                if tagdict ["key"] == "postcode":
                    tagdict['value'] = zip 
                    
                    
                #If a value for city was included in the osm file, value corrected above added
                # to dictionary.  
                if tagdict["key"] == "city":
                    tagdict ["value"] = city
            
                
                # Funeral directors corrected as "amenity" key.
                if tagdict["key"] == "shop" and tagdict["value"] == "funeral_directors":
                    tagdict["key"] == "amenity"

                # Ice_cream and laundry corrected as "shop" key.
                if tagdict["key"] == "amenity" and (tagdict["value"] == "ice_cream" or tagdict["value"] == "laundry"):
                    tagdict["key"] = "shop"

                #Add all remaining tags to final dictionary 
                waytags.append(tagdict)
        
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': waytags}

# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)
#Isolates street type to pass into streetmapping below.

streetmapping = {
"Ave" : "Avenue",
"Ave." : "Avenue",
"Avene" : "Avenue",
"Avenue," : "Avenue",
"avenue" : "Avenue",
"ave": "Avenue",
"brway":"Broadway",
"Blvd" :"Boulevard",
"Ctr": "Center",
"Dr":"Drive",
"Plz" : "Plaza",
"Rd" : "Road",
"Sq" : "Square",
"ST" : "Street",
'St': "Street",
'St.' : "Street",
 "St.," : "Street",   
'Steet' : "Street",
'st' : "Street",
'street' : "Street",
"Street,":"Street"
}

allstreettypes = []
# Creates a list of all street ending to be used to seperate suite value.
for k, v in streetmapping.items():
    allstreettypes.append(k)
    allstreettypes.append (v)



def correctstreet (streetname):
    '''Corrects street names for abbreviations and also seperates out suites/unit numbers.'''
    m = street_type_re.search(streetname)
    if m:
        street_type = m.group()
        numerical_street_type = re.search(r'\d', street_type)
        # Seperates out street addresses ending in suite or unit numbers
        if numerical_street_type:
            streetname, suite = numerical (streetname)
            
        # If non-numerical street ending, but street type is an abbreviation:
        elif street_type in streetmapping:
            corrected_street_type = streetmapping[street_type]
            streetname = street_type_re.sub(corrected_street_type, streetname)
            suite = "NA"
        
        # If neither of the above, nothing is changed.
        else:
            streetname = streetname
            suite = "NA"
        
        return streetname, suite

def numerical (streetname):
    '''If street ends in numerical string, this fuction seperates a suite/unit address
        where applicable'''
    
    streetname = streetname.split(" ")
    
    #Finds where street type occurs in string 
    if bool(set(streetname) & set(allstreettypes)):
        for i in streetname:
            if i in allstreettypes:
                #Creates a value for suite/unit number after street type.
                pos = streetname.index(i)
                suite = streetname [pos+1:]
                suite = " ".join(suite)
                #Creates a value for street address up to and including street type.
                streetname = streetname[:pos]
                streetname = " ".join(streetname)
                streetname = streetname + " " + i
                
                #Corrects the new value street type if abbreviated.
                m = street_type_re.search(streetname)
                if m:
                    street_type = m.group()
                    if street_type in streetmapping:
                        corrected_street_type = streetmapping[street_type]
                        streetname = street_type_re.sub(corrected_street_type, streetname)
                    else:
                        streetname = streetname
               
                return streetname, suite
                
    else:
         #If no street type in streetname, ex. street name ends in  all numbers, creates
         # a non Null value for suite
        streetname = " ".join(streetname) 
        suite = "NA"

        return streetname, suite    
        
def is_number(s):
    ''' Checks whether argument contains a numerical character. 
    http://stackoverflow.com/questions/31083503/how-do-i-check-if-a-string-contains-any-numbers'''
    if any(str.isdigit(c) for c in s):
        return True
    else:
        return False
    
    
def correctzip (s):
    '''Audits zip code for consistent format'''
    s = re.sub("\D", "", s) 
    s = s[0:5]
    #Old World Trade Center zip discontinued
    if s == "10048":
        s = "10007"
    return s

def correctcity (city):
    '''Audits city for consistent format'''
    if city in citymapping:
        city = citymapping[city]
    return city

citymapping = {
            "Brookklyn": "Brooklyn",
           "brooklyn": "Brooklyn", 
           "Brooklyn, NY": "Brooklyn",
           "Brooklyn, New York": "Brooklyn", 
           "Brooklyn ": "Brooklyn",
            "M": "New York",
           "Manhattan NYC": "New York",
           "NEW YORK CITY": "New York",
           "New York CIty": "New York", 
           "New York City": "New York",
           "New York, NY": "New York",
           "Tribeca": "New York",
           "York City": "New York",
           "new york": "New York",
           "Glendale, NY":"Glendale",
           "Ozone Park, NY": "Ozone Park",
            "queens": "Queens"
}

# Where a zip code is given, this dictionary checks correct city is also added.
zipcodemapping = {'10017':["New York"],'10013': ['New York'], '11231': ['Brooklyn'], '10011': ['New York'], '10010': ['New York'], '10016': ['New York'], '10014': ['New York'], '10282': ['New York'], '10280': ['New York'], '10281': ['New York'], '11379': ['Middle Village'], '11378': ['Maspeth'], '11375': ['Forest Hills'], '11374': ['Rego Park'], '11377': ['Woodside'], '11101': ['Blissville'], '11373': ['Elhhurst'], '71877': ['Brooklyn'], '10045': ['New York'], '11201': ['Brooklyn'], '11203': ['Brooklyn'], '11205': ['Brooklyn'], '11204': ['Brooklyn'], '11207': ['Brooklyn'], '11206': ['Brooklyn'], '11209': ['Brooklyn'], '11208': ['Brooklyn'], '11104': ['Sunnyside'], '11368': ['Corona'], '11367': ['Flushing'], '11222': ['Brooklyn'], '07030': ['Hoboken'], '11212': ['Brooklyn'], '07302': ['Jersey City'], '11210': ['Brooklyn'], '11211': ['Brooklyn'], '11216': ['Brooklyn'], '07306': ['Jersey City'], '11214': ['Brooklyn'], '11215': ['Brooklyn'], '11218': ['Brooklyn'], '11219': ['Brooklyn'], '11213': ['Brooklyn'], '11697': ['Breezy Point'], '11694': ['Rockaway Park'], '10023': ['New York'], '10038': ['New York'], '10275': ['New York'], '11229': ['Brooklyn'], '11228': ['Brooklyn'], '11226': ['Brooklyn'], '11225': ['Brooklyn'], '11224': ['Brooklyn'], '11223': ['Brooklyn'], '07311': ['Jersey City'], '11221': ['Brooklyn'], '11220': ['Brooklyn'], '11421': ['Woodhaven'], '11420': ['South Ozone Park'], '11238': ['Brooklyn'], '11239': ['Brooklyn'], '10018': ['New York'], '11230': ['Brooklyn'], '10012': ['New York'], '11232': ['Brooklyn'], '11233': ['Brooklyn'], '11234': ['Brooklyn'], '11235': ['Brooklyn'], '11236': ['Brooklyn'], '11237': ['Brooklyn'], '10004': ['New York'], '10005': ['New York'], '10006': ['New York'], '10007': ['New York'], '10001': ['New York'], '10002': ['New York'], '10003': ['New York'], '10009': ['New York'], '07310': ['Jersey City'], '11249': ['Brooklyn'], '11217': ['Brooklyn'], '11414': ['Queens'], '11415': ['Kew Gardens'], '11416': ['Ozone Park'], '11417': ['Ozone Park'], '11418': ['Richmond Hill'], '11419': ['South Richmond Hill'], '11251': ['Brooklyn']}
# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file,          codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file,          codecs.open(WAYS_PATH, 'w') as ways_file,          codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file,          codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
     #Note: Validation is ~ 10X slower. For the project consider using a small
     #sample of the map when validating.
   process_map(OSM_PATH, validate=False)


# In[ ]:




# In[ ]:



