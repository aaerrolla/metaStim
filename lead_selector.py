import csv

from lead import Lead

class LeadSelector():
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.leads = self.load_leads()

    def load_leads(self):
        leads = {}
        with open(self.csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
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
                leads[lead_id] = Lead(lead_id, lead_type, company, no, he, re, ae, ies, htip)
        return leads   

    def select_lead(self, id):
        return self.leads.get(id)

