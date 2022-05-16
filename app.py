#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import LabelEncoder


# In[2]:


model = joblib.load('model.pkl')
app = Flask(__name__)


# In[3]:


regions = ['birmingham', 'huntsville / decatur', 'dothan', 'mobile',
       'montgomery', 'florence / muscle shoals', 'gadsden-anniston',
       'tuscaloosa', 'anchorage / mat-su', 'fairbanks', 'phoenix',
       'flagstaff / sedona', 'tucson', 'little rock', 'prescott', 'yuma',
       'fayetteville', 'bakersfield', 'texarkana', 'fresno / madera',
       'hanford-corcoran', 'inland empire', 'los angeles',
       'kenai peninsula', 'southeast alaska', 'mohave county',
       'fort smith', 'jonesboro', 'show low', 'gold country',
       'sierra vista', 'humboldt county', 'chico', 'imperial county',
       'boulder', 'modesto', 'orange county', 'mendocino county',
       'merced', 'palm springs', 'reno / tahoe', 'monterey bay',
       'redding', 'sacramento', 'san diego', 'san luis obispo',
       'santa barbara', 'SF bay area', 'stockton', 'ventura county',
       'susanville', 'colorado springs', 'yuba-sutter', 'visalia-tulare',
       'siskiyou county', 'santa maria', 'denver', 'northwest CT',
       'fort collins / north CO', 'eastern CT', 'hartford', 'new haven',
       'western slope', 'washington, DC', 'pueblo', 'daytona beach',
       'high rockies', 'ft myers / SW florida', 'gainesville',
       'jacksonville', 'ocala', 'eastern CO', 'delaware', 'lakeland',
       'florida keys', 'north central FL', 'heartland florida', 'orlando',
       'panama city', 'pensacola', 'sarasota-bradenton', 'south florida',
       'okaloosa / walton', 'space coast', 'tallahassee',
       'tampa bay area', 'atlanta', 'augusta', 'brunswick', 'athens',
       'albany', 'treasure coast', 'st augustine',
       'macon / warner robins', 'columbus', 'northwest GA',
       'savannah / hinesville', 'hawaii', 'boise', 'east idaho',
       'statesboro', "spokane / coeur d'alene", 'valdosta',
       'bloomington-normal', 'chicago', 'twin falls', 'decatur',
       'la salle co', 'quad cities, IA/IL', 'st louis, MO', 'peoria',
       'evansville', 'springfield', 'bloomington', 'lewiston / clarkston',
       'pullman / moscow', 'indianapolis', 'fort wayne', 'rockford',
       'south bend / michiana', 'ames', 'richmond', 'champaign urbana',
       'southern illinois', 'mattoon-charleston', 'muncie / anderson',
       'western IL', 'lafayette / west lafayette', 'kokomo',
       'terre haute', 'cedar rapids', 'des moines',
       'omaha / council bluffs', 'wichita', 'fort dodge', 'lawrence',
       'bowling green', 'lexington', 'eastern kentucky', 'iowa city',
       'louisville', 'dubuque', 'baton rouge', 'waterloo / cedar falls',
       'manhattan', 'western KY', 'lafayette', 'monroe', 'mason city',
       'new orleans', 'kansas city, MO', 'southeast IA', 'topeka',
       'lake charles', 'southeast KS', 'sioux city', 'huntington-ashland',
       'salina', 'northwest KS', 'southwest KS', 'central louisiana',
       'houma', 'owensboro', 'shreveport', 'lansing', 'maine',
       'annapolis', 'frederick', 'southern maryland', 'boston',
       'south coast', 'western massachusetts', 'worcester / central MA',
       'baltimore', 'ann arbor', 'battle creek', 'detroit metro',
       'holland', 'flint', 'kalamazoo', 'western maryland', 'muskegon',
       'saginaw-midland-baycity', 'upper peninsula', 'eastern shore',
       'cumberland valley', 'cape cod / islands', 'grand rapids',
       'bemidji', 'central michigan', 'northern michigan', 'jackson',
       'southwest michigan', 'the thumb', 'port huron', 'brainerd',
       'duluth / superior', 'fargo / moorhead', 'minneapolis / st paul',
       'rochester', 'gulfport / biloxi', 'st cloud', 'hattiesburg',
       'asheville', 'north mississippi', 'kirksville', 'mankato',
       'southwest MS', 'columbia / jeff city', 'southwest MN',
       'st joseph', 'charlotte', 'boone', 'billings', 'missoula',
       'eastern NC', 'greensboro', 'raleigh / durham / CH',
       'hickory / lenoir', 'lake of the ozarks', 'meridian', 'joplin',
       'southeast missouri', 'kansas city', 'st louis', 'bozeman',
       'eastern montana', 'kalispell', 'helena', 'butte', 'great falls',
       'outer banks', 'wilmington', 'winston-salem', 'lincoln',
       'las vegas', 'north platte', 'north jersey', 'jersey shore',
       'south jersey', 'central NJ', 'albuquerque', 'buffalo', 'ithaca',
       'farmington', 'syracuse', 'watertown', 'hudson valley',
       'grand island', 'santa fe / taos', 'long island', 'new york city',
       'binghamton', 'scottsbluff / panhandle', 'plattsburgh-adirondacks',
       'utica-rome-oneida', 'new hampshire', 'elko', 'clovis / portales',
       'finger lakes', 'catskills', 'chautauqua', 'elmira-corning',
       'las cruces', 'roswell / carlsbad', 'glens falls',
       'potsdam-canton-massena', 'oneonta', 'twin tiers NY/PA',
       'cincinnati', 'akron / canton', 'bismarck', 'grand forks',
       'dayton / springfield', 'cleveland', 'lima / findlay', 'toledo',
       'northern panhandle', 'north dakota', 'oklahoma city',
       'zanesville / cambridge', 'ashtabula', 'lawton', 'chillicothe',
       'texoma', 'tulsa', 'bend', 'corvallis/albany', 'tuscarawas co',
       'east oregon', 'eugene', 'medford-ashland', 'oregon coast',
       'portland', 'mansfield', 'stillwater', 'parkersburg-marietta',
       'northwest OK']
parking_options = ['off-street parking', 'street parking', 'carport',
       'attached garage', 'detached garage', 'no parking',
       'valet parking']
laundry_options = ['laundry on site', 'w/d in unit', 'w/d hookups', 'laundry in bldg',
       'no laundry on site']
states = ['al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'dc', 'fl', 'de', 'ga',
       'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'mi', 'me', 'md',
       'ma', 'mn', 'ms', 'nc', 'mo', 'mt', 'ne', 'nv', 'nj', 'nm', 'ny',
       'nh', 'oh', 'nd', 'ok', 'or']
types = ['apartment', 'house', 'manufactured', 'townhouse', 'condo',
       'duplex', 'in-law', 'cottage/cabin', 'flat', 'loft', 'land',
       'assisted living']


# In[4]:


us_state_to_abbrev = {
"Alabama": "AL",
"Alaska": "AK",
"Arizona": "AZ",
"Arkansas": "AR",
"California": "CA",
"Colorado": "CO",
"Connecticut": "CT",
"Delaware": "DE",
"Florida": "FL",
"Georgia": "GA",
"Hawaii": "HI",
"Idaho": "ID",
"Illinois": "IL",
"Indiana": "IN",
"Iowa": "IA",
"Kansas": "KS",
"Kentucky": "KY",
"Louisiana": "LA",
"Maine": "ME",
"Maryland": "MD",
"Massachusetts": "MA",
"Michigan": "MI",
"Minnesota": "MN",
"Mississippi": "MS",
"Missouri": "MO",
"Montana": "MT",
"Nebraska": "NE",
"Nevada": "NV",
"New Hampshire": "NH",
"New Jersey": "NJ",
"New Mexico": "NM",
"New York": "NY",
"North Carolina": "NC",
"North Dakota": "ND",
"Ohio": "OH",
"Oklahoma": "OK",
"Oregon": "OR",
"Pennsylvania": "PA",
"Rhode Island": "RI",
"South Carolina": "SC",
"South Dakota": "SD",
"Tennessee": "TN",
"Texas": "TX",
"Utah": "UT",
"Vermont": "VT",
"Virginia": "VA",
"Washington": "WA",
"West Virginia": "WV",
"Wisconsin": "WI",
"Wyoming": "WY",
"District of Columbia": "DC",
"American Samoa": "AS",
"Guam": "GU",
"Northern Mariana Islands": "MP",
"Puerto Rico": "PR",
"United States Minor Outlying Islands": "UM",
"U.S. Virgin Islands": "VI",
}


# In[5]:


def lower_dict(d):
   new_dict = dict((k, v.lower()) for k, v in d.items())
   return new_dict


# In[6]:


us_state_to_abbrev = lower_dict(us_state_to_abbrev)
us_state_to_abbrev = dict((v,k) for k,v in us_state_to_abbrev.items())
us_state_to_abbrev


# In[7]:


temp = []
for i in range(len(states)):
    temp.append(us_state_to_abbrev[states[i]])
states = temp
states


# In[8]:


@app.route("/")
def home():
    return render_template("home.html",regions=regions,states=states,house_types=types,laundry_options=laundry_options,parking_options=parking_options)


# In[9]:


@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        region = request.form['region']
        house_type = request.form['type'] 
        sqfeet = request.form['area'] #Integer between 169 and 1722
        beds = request.form['beds'] #Integer between 0 and 3
        baths = request.form['baths'] #Integer between 0.0 and 0.3 with step=0.5
        cats_allowed_option = request.form['cats_allowed']
        cats_allowed = 0
        if cats_allowed_option == "YES":
            cats_allowed = 1
        elif cats_allowed_option == "NO":
            cats_allowed = 0
        dogs_allowed_option = request.form['dogs_allowed']
        dogs_allowed = 0
        if dogs_allowed_option == "YES":
            dogs_allowed = 1
        elif dogs_allowed_option == "NO":
            dogs_allowed = 0
        smoking_allowed_option = request.form['smoking_allowed']
        smoking_allowed = 0
        if smoking_allowed_option == "YES":
            smoking_allowed = 1
        elif smoking_allowed_option == "NO":
            smoking_allowed = 0
        wheelchairaccess_option = request.form['wheelchair_access']
        wheelchairaccess = 0
        if wheelchairaccess_option == "YES":
            wheelchairaccess = 1
        elif wheelchairaccess_option == "NO":
            wheelchairaccess = 0
        elec_vehicle_charge_option = request.form['electric_vehicle_charge']
        electric_vehicle_charge = 0
        if elec_vehicle_charge_option == "YES":
            electric_vehicle_charge = 1
        elif elec_vehicle_charge_option == "NO":
            electric_vehicle_charge = 0
        furnished_status = request.form['furnished']
        is_furnished = 0
        if furnished_status == "YES":
            is_furnished = 1
        elif furnished_status == "NO":
            is_furnished = 0
        laundry_options_select = request.form['laundry']
        le = LabelEncoder()
        laundry_options = le.fit_transform([laundry_options_select])
        parking_options_select = request.form['parking']
        parking_options = le.fit_transform([parking_options_select])
        region = le.fit_transform([region])
        house_type = le.fit_transform([house_type])
        lat = request.form['latitude'] #Range:(-11.04849086221126,11.514647198246161)
        long = request.form['longitude'] #Range:(-4.283912334222467,11.052201259378677)
        state_select = request.form['states']
        state = le.fit_transform([state_select])
        
        predictions = model.predict([[
            region,
            house_type,
            sqfeet,
            beds,
            baths,
            cats_allowed,
            dogs_allowed,
            smoking_allowed,
            wheelchairaccess,
            electric_vehicle_charge,
            is_furnished,
            laundry_options,
            parking_options,
            lat,
            long,
            state
        ]])
        
        output = predictions[0]
        return render_template("home.html",prediction_text="Your predicted house price is ${}.".format("%.2f"% output),regions=regions,states=states,house_types=types,laundry_options=laundry_options,parking_options=parking_options)


# In[ ]:


if __name__ == "__main__":
    app.run(port=8080)

