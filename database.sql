CREATE TABLE IF NOT EXISTS Cats (
    Id INTEGER PRIMARY KEY,
    Name TEXT NOT NULL,
    Sex TEXT NOT NULL,
    Bonded BOOLEAN NOT NULL,
    Added DATETIME NOT NULL,
    Birthday DATETIME NOT NULL,
    Dog_Friendliness TEXT NOT NULL,
    Cat_Friendliness TEXT NOT NULL,
    Breed TEXT NOT NULL,
    Status TEXT NOT NULL,
    Image TEXT NOT NULL,
    Analyzed BOOLEAN NOT NULL,
    Notified BOOLEAN NOT NULL
);
