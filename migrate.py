import asyncio
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

load_dotenv()

async def migrate():
    engine = create_async_engine(os.getenv("DATABASE_URL"))
    
    async with engine.begin() as conn:

        # 1. Créer la table sessions
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sessions (
                id SERIAL PRIMARY KEY,
                phone VARCHAR(20) NOT NULL,
                statut VARCHAR(20) DEFAULT 'HUMAIN',
                agent VARCHAR(100),
                agent_cloture VARCHAR(100),
                cree_le TIMESTAMP DEFAULT NOW(),
                prise_en_charge_le TIMESTAMP,
                premiere_reponse_le TIMESTAMP,
                cloturee_le TIMESTAMP,
                problematique VARCHAR(200),
                commentaire_agent TEXT,
                note_client INTEGER,
                en_attente_note BOOLEAN DEFAULT FALSE
            );
        """))

        # 2. Ajouter session_id dans messages
        await conn.execute(text("""
            ALTER TABLE messages
            ADD COLUMN IF NOT EXISTS session_id INTEGER REFERENCES sessions(id);
        """))

        # 3. Ajouter nom_agent dans messages
        await conn.execute(text("""
            ALTER TABLE messages
            ADD COLUMN IF NOT EXISTS nom_agent VARCHAR(100);
        """))

        # 4. Supprimer les colonnes devenues inutiles dans conversations
        await conn.execute(text("""
            ALTER TABLE conversations
            DROP COLUMN IF EXISTS agent;
        """))
        await conn.execute(text("""
            ALTER TABLE conversations
            DROP COLUMN IF EXISTS agent_cloture;
        """))
        await conn.execute(text("""
            ALTER TABLE conversations
            DROP COLUMN IF EXISTS prise_en_charge_le;
        """))
        await conn.execute(text("""
            ALTER TABLE conversations
            DROP COLUMN IF EXISTS premiere_reponse_le;
        """))
        await conn.execute(text("""
            ALTER TABLE conversations
            DROP COLUMN IF EXISTS cloturee_le;
        """))
        await conn.execute(text("""
            ALTER TABLE conversations
            DROP COLUMN IF EXISTS problematique;
        """))
        await conn.execute(text("""
            ALTER TABLE conversations
            DROP COLUMN IF EXISTS commentaire_agent;
        """))
        await conn.execute(text("""
            ALTER TABLE conversations
            DROP COLUMN IF EXISTS note_client;
        """))
        await conn.execute(text("""
            ALTER TABLE conversations
            DROP COLUMN IF EXISTS en_attente_note;
        """))
        await conn.execute(text("""
            ALTER TABLE conversations
            DROP COLUMN IF EXISTS mis_a_jour_le;
        """))
        await conn.execute(text("""
            ALTER TABLE conversations
            DROP COLUMN IF EXISTS session_courante;
        """))

        # 5. Ajouter problematiques si pas encore créée
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS problematiques (
                id SERIAL PRIMARY KEY,
                libelle VARCHAR(200) UNIQUE NOT NULL,
                cree_le TIMESTAMP DEFAULT NOW()
            );
        """))

        # 6. Ajouter messages_ia si pas encore créée
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages_ia (
                id         SERIAL PRIMARY KEY,
                phone      VARCHAR(20) NOT NULL,
                question   TEXT NOT NULL,
                reponse    TEXT NOT NULL,
                sources    TEXT,
                duree_ms   INTEGER,
                a_handover BOOLEAN DEFAULT FALSE,
                envoye_le  TIMESTAMP DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_messages_ia_phone 
                ON messages_ia(phone);
            CREATE INDEX IF NOT EXISTS idx_messages_ia_envoye_le 
                ON messages_ia(envoye_le);
        """))
        await conn.execute(text("""
            ALTER TABLE messages_ia 
            ADD COLUMN IF NOT EXISTS problematique_id INTEGER 
                REFERENCES problematiques(id),
            ADD COLUMN IF NOT EXISTS problematique_libelle VARCHAR(200);
        """))

        # 7. Ajouter sessions_ia si pas encore créée
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sessions_ia (
                id                    SERIAL PRIMARY KEY,
                phone                 VARCHAR(20) NOT NULL,
                statut                VARCHAR(20) DEFAULT 'EN_COURS',
                problematique_id      INTEGER REFERENCES problematiques(id),
                problematique_libelle VARCHAR(200),
                nb_echanges           INTEGER DEFAULT 0,
                debut_le              TIMESTAMP DEFAULT NOW(),
                cloturee_le           TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_ia_phone 
                ON sessions_ia(phone);

        
            CREATE TABLE IF NOT EXISTS messages_session_ia (
                id         SERIAL PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES sessions_ia(id),
                phone      VARCHAR(20) NOT NULL,
                expediteur VARCHAR(10) NOT NULL,
                texte      TEXT NOT NULL,
                duree_ms   INTEGER,
                envoye_le  TIMESTAMP DEFAULT NOW()
            );
        """))


        print("✅ Table messages_ia créée")

asyncio.run(migrate())