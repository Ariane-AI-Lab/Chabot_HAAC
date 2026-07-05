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


class Problematique(Base):
    __tablename__ = "problematiques"

    id      = Column(Integer, primary_key=True, autoincrement=True)
    libelle = Column(String(200), unique=True, nullable=False)
    cree_le = Column(DateTime, server_default=func.now())