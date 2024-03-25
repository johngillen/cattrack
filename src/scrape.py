from bs4 import BeautifulSoup as soup
from datetime import date
import regex
import asyncio
import aiohttp
from src.database import cat
from src.config import config

parsedate = lambda s: date(int(s.split('/')[2]), int(s.split('/')[0]), int(s.split('/')[1]))

def fetch_page(url):
    async def fetch(url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()
    return soup(asyncio.run(fetch(url)), 'html.parser')

def parse_cat(catdata):
    id = int(catdata['9'].split('-')[1])
    name = catdata['1']
    sex = catdata['20']
    bonded = catdata['4'].lower().count('bonded') > 0
    added = parsedate(catdata['2'])
    birthday = parsedate(catdata['22'])
    if 'Dogs:' in catdata['130']: dog_friendliness = regex.search(r'Dogs: (.+?)\n', catdata['130']).group(1)
    else: dog_friendliness = 'Unknown'
    if 'Cats:' in catdata['130']: cat_friendliness = regex.search(r'Cats: (.+?)\n', catdata['130']).group(1)
    else: cat_friendliness = 'Unknown'
    if '6' in catdata: breed = catdata['6']
    else: breed = 'Unknown'
    status = catdata['19']
    image = catdata['9']
    return cat(id, name, sex, bonded, added, birthday, dog_friendliness, cat_friendliness, breed, status, image)

def fetch_cats(page = fetch_page(f'{config["url"]}/cats')):
    cats = []
    catlist = page.select('div.availablePetHidden')
    for catdata in catlist:
        k = catdata.text.replace('||', '|').split('|')[1:-1:2]
        v = catdata.text.replace('||', '|').split('|')[2:-1:2]
        cat = parse_cat(dict(zip(k, v)))
        cats.append(cat)
    return cats

if __name__ == '__main__':
    cats = fetch_cats()
    for cat in cats:
        print(cat)
        cat.save()
