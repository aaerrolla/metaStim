import csv
from importlib import resources

from metastim import lead

class LeadSelector():
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.leads = self.load_leads()

    def load_leads(self):
        leads = {}
        with resources.open_text("metastim", self.csv_file) as csv_resource:
            with open(csv_resource.name) as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    lead_id = row['Lead']
                    lead_type = row['Type']
                    company = row['Company']
                    no = int(row['no'])
                    he = float(row['h_e [mm]'])
                    re = float(row['r_e [mm]'])
                    ae = float(row['a_e [deg]']) if row['a_e [deg]'] != 'N/A' else None
                    ies = float(row['ies [mm]'])
                    htip = float(row['h_tip [mm]'])
                    leads[lead_id] = lead.Lead(lead_id, lead_type, company, no, he, re, ae, ies, htip)
        return leads   

    def select_lead(self, id):
        return self.leads.get(id)

