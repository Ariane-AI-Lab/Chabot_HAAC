from sqlalchemy import Column, String, Text, Boolean, DateTime, Integer, ForeignKey
from sqlalchemy.sql import func
from database import Base


class Agent(Base):
    __tablename__ = "agents"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    nom          = Column(String(100), nullable=False)
    email        = Column(String(200), unique=True, nullable=False)
    mot_de_passe = Column(Text, nullable=False)
    role         = Column(String(20), default="agent")
    actif        = Column(Boolean, default=True)
    cree_le      = Column(DateTime, server_default=func.now())


class Conversation(Base):
    """Historique global par numéro WhatsApp — table légère."""
    __tablename__ = "conversations"

    phone   = Column(String(20), primary_key=True)
    statut  = Column(String(20), default="IA")
    cree_le = Column(DateTime, server_default=func.now())


class Session(Base):
    """Une session = un handover distinct pour un numéro donné."""
    __tablename__ = "sessions"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    phone               = Column(String(20), nullable=False)
    statut              = Column(String(20), default="HUMAIN")
    agent               = Column(String(100), nullable=True)
    agent_cloture       = Column(String(100), nullable=True)
    cree_le             = Column(DateTime, server_default=func.now())
    prise_en_charge_le  = Column(DateTime, nullable=True)
    premiere_reponse_le = Column(DateTime, nullable=True)
    cloturee_le         = Column(DateTime, nullable=True)
    problematique       = Column(String(200), nullable=True)
    commentaire_agent   = Column(Text, nullable=True)
    note_client         = Column(Integer, nullable=True)
    en_attente_note     = Column(Boolean, default=False)


class Message(Base):
    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    phone      = Column(String(20), nullable=False)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=True)
    expediteur = Column(String(20), nullable=False)
    texte      = Column(Text, nullable=False)
    nom_agent  = Column(String(100), nullable=True)
    envoye_le  = Column(DateTime, server_default=func.now())

class MessageIA(Base):
    __tablename__ = "messages_ia"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    phone           = Column(String(20), nullable=False, index=True)
    question        = Column(Text, nullable=False)
    reponse         = Column(Text, nullable=False)
    sources         = Column(Text, nullable=True)
    duree_ms        = Column(Integer, nullable=True)
    a_handover      = Column(Boolean, default=False)
    problematique_id = Column(Integer, ForeignKey("problematiques.id"), nullable=True)
    problematique_libelle = Column(String(200), nullable=True)  # dénormalisé pour simplicité
    envoye_le       = Column(DateTime, server_default=func.now())


class MessageSessionIA(Base):
    __tablename__ = "messages_session_ia"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(Integer, ForeignKey("sessions_ia.id"), nullable=False)
    phone       = Column(String(20), nullable=False)
    expediteur  = Column(String(10), nullable=False)  # "client" ou "ia"
    texte       = Column(Text, nullable=False)
    duree_ms    = Column(Integer, nullable=True)
    envoye_le   = Column(DateTime, server_default=func.now())


class SessionIA(Base):
    __tablename__ = "sessions_ia"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    phone                 = Column(String(20), nullable=False, index=True)
    statut                = Column(String(20), default="EN_COURS")
    # "EN_COURS" | "CLOTUREE_AU_REVOIR" | "CLOTUREE_TIMEOUT"
    problematique_id      = Column(Integer, ForeignKey("problematiques.id"), nullable=True)
    problematique_libelle = Column(String(200), nullable=True)
    nb_echanges           = Column(Integer, default=0)
    debut_le              = Column(DateTime, server_default=func.now())
    cloturee_le           = Column(DateTime, nullable=True)


class Problematique(Base):
    __tablename__ = "problematiques"

    id      = Column(Integer, primary_key=True, autoincrement=True)
    libelle = Column(String(200), unique=True, nullable=False)
    cree_le = Column(DateTime, server_default=func.now())