# class Represents Lead

class Lead:
    # initializr
    def __init__(self, lead_id = "6172", type = "Directional", company = "Abbot Laboratories", no = 8, he = 1.5, re = 0.635, ae = 104, ies = 0.5, htip = 1.5):
        self.lead_id = lead_id
        self.type = type
        self.company = company
        self.no = no
        self.he = he
        self.re = re
        self.ae = ae
        self.ies = ies
        self.htip = htip
    
    def __str__(self):
        return f"Lead: {self.lead_id}, Lead Type: {self.type}, Company: {self.company}, No: {self.no}, he : {self.he}, re : {self.re}, ae : {self.ae}, ies: {self.ies}, htip : {self.htip}"

    def __repr__(self):
        return f"Lead(lead_id='{self.lead_id}', type='{self.type}', company='{self.company}', no={self.no}, he={self.he}, re={self.re}, ae={self.ae}, ies={self.ies}, htip={self.htip})"