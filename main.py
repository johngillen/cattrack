from src import classify, database, message, scrape
from src.config import config
import os, asyncio, datetime
from tqdm import tqdm

def main():
    context = database.context()
    classifier = classify.classifier()

    def cat_criteria(cat: database.cat) -> bool:
        return cat.status == 'Available' and \
               bool(cat.bonded) == False and \
               any(set(classifier.classify_topn(f'catche/{cat.id}.jpg', 3)) & set(config['breeds']))

    web_cats = scrape.fetch_cats()
    for cat in tqdm(web_cats):
        if context.cat_exists(cat.id):
            context.update_cat(cat)
        else:
            context.add_cat(cat)

    db_cats = context.get_all_cats()
    for cat in tqdm(db_cats):
        if cat.id not in [web_cat.id for web_cat in web_cats]:
            context.delete_cat(cat)
            os.remove(f'images/{cat.id}.jpg')
            continue

        if not cat.analyzed:
            cat.fetch_image()
            cat.breed = classifier.classify(f'catche/{cat.id}.jpg')
            cat.analyzed = True
            context.update_cat(cat)

        if not cat_criteria(cat): continue

        if not cat.notified:
            msg = f'''\
{cat.name} is available for adoption!
sex: {cat.sex[0]}
breed(s)?: {', '.join(classifier.classify_topn(f'catche/{cat.id}.jpg', 3))}
age: {(datetime.datetime.now() - datetime.datetime.fromisoformat(cat.birthday)).days / 365:.1f} years
info: {config['url']}/adoptable-animals-details?id={cat.id}
'''
            message.send(msg)
            cat.notified = True
            context.update_cat(cat)

    context.close_connection()

if __name__ == '__main__': main()
