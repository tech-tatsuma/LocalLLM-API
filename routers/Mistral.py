from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing import Dict
import csv
import os

from .functions.retrieval import get_answer_with_search

router = APIRouter(
    prefix="/mistral",
    tags=["mistral"],
)

@router.post("/search", response_model=Dict[str, str]):
def askwithsearch(question: str):
    return get_answer_with_search(question)