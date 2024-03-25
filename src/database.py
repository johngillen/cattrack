import sqlite3
from datetime import date
import aiohttp
from src.config import config
from contextlib import closing
import requests


class cat:
    id = 0
    name = ""
    sex = ""
    bonded = False
    added = date(1970, 1, 1)
    birthday = date(1970, 1, 1)
    dog_friendliness = ""
    cat_friendliness = ""
    breed = ""
    status = ""
    image = ""
    analyzed = False
    notified = False

    def __init__(self, id, name, sex, bonded, added, birthday, dog_friendliness, cat_friendliness, breed, status, image, analyzed = False, notified = False):
        self.id = id
        self.name = name
        self.sex = sex
        self.bonded = bonded
        self.added = added
        self.birthday = birthday
        self.dog_friendliness = dog_friendliness
        self.cat_friendliness = cat_friendliness
        self.breed = breed
        self.status = status
        self.image = image
        self.analyzed = analyzed
        self.notified = notified

    async def async_fetch_image(self):
        url = f'{config["url"]}//media/additionalForms/{self.image}'
        async def fetch_image(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.read()
        with open(f'catche/{self.id}.jpg', 'w+b') as f:
            if f.read() == b'': f.write(await fetch_image(url))

    def fetch_image(self):
        url = f'{config["url"]}//media/additionalForms/{self.image}'
        with open(f'catche/{self.id}.jpg', 'w+b') as f:
            if f.read() == b'': f.write(requests.get(url).content)

class context:
    con = None

    def __init__(self):
        self.create_connection()
        self.create_table()

    def create_connection(this):
        this.con = sqlite3.connect('database.sqlite3')

    def close_connection(this):
        this.con.close()

    def create_table(this):
        sql_create_table = open('database.sql', 'r').read()
        this.con.execute(sql_create_table)

    def add_cat(this, cat: cat):
        if this.cat_exists(cat.id): return

        sql = ''' INSERT INTO Cats(Id, Name, Sex, Bonded, Added, Birthday, Dog_Friendliness, Cat_Friendliness, Breed, Status, Image, Analyzed, Notified)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) '''
        with closing(this.con.cursor()) as cur:
            cur.execute(sql, (cat.id, cat.name, cat.sex, cat.bonded, cat.added, cat.birthday, cat.dog_friendliness, cat.cat_friendliness, cat.breed, cat.status, cat.image, cat.analyzed, cat.notified))
            this.con.commit()
            return cur.lastrowid

    def get_all_cats(this):
        with closing(this.con.cursor()) as cur:
            cur.execute('SELECT * FROM Cats')
            rows = cur.fetchall()
            return [cat(*row) for row in rows]

    def get_cat(this, id):
        with closing(this.con.cursor()) as cur:
            cur.execute('SELECT * FROM Cats WHERE Id=?', (id,))
            rows = cur.fetchall()
            return cat(*rows[0])

    def delete_cat(this, id):
        if not this.cat_exists(id): return

        sql = 'DELETE FROM Cats WHERE Id=?'
        with closing(this.con.cursor()) as cur:
            cur = this.con.cursor()
            cur.execute(sql, (id,))
            this.con.commit()

    def update_cat(this, cat: cat):
        if not this.cat_exists(cat.id): return
        
        sql = '''
        UPDATE Cats
        SET Name = ?, Sex = ?, Bonded = ?, Added = ?, Birthday = ?, Dog_Friendliness = ?, Cat_Friendliness = ?, Breed = ?, Status = ?, Image = ?, Analyzed = ?, Notified = ?
        WHERE Id = ?
        '''
        
        with closing(this.con.cursor()) as cur:
            cur.execute(sql, (cat.name, cat.sex, cat.bonded, cat.added, cat.birthday, cat.dog_friendliness, cat.cat_friendliness, cat.breed, cat.status, cat.image, cat.analyzed, cat.notified, cat.id))
            this.con.commit()

    def cat_exists(this, id):
        with closing(this.con.cursor()) as cur:
            cur.execute('SELECT * FROM Cats WHERE Id=?', (id,))
            rows = cur.fetchall()
            return len(rows) > 0
