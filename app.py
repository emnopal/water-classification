import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, BaseConfig
from typing import Union

from starlette.responses import RedirectResponse

app = FastAPI(debug=True)


class Params(BaseModel):
    ph: Union[float, str]
    Hardness: Union[float, str]
    Solids: Union[float, str]
    Chloramines: Union[float, str]
    Sulfate: Union[float, str]
    Conductivity: Union[float, str]
    OrganicCarbon: Union[float, str]
    Trihalomethanes: Union[float, str]
    Turbidity: Union[float, str]


@app.get('/api/v1/')
async def docs():
    return RedirectResponse('/openapi.json')


@app.post('/api/v1/')
async def water_classification(
        params: Params,
        model: str = None,
) -> dict:
    params_ = np.array(list(params.dict().values()))

    print(params_)

    if model:

        if model.lower() == 'decision_tree':
            model_ = joblib.load('model/decision_tree')
            result = model_.predict(params_.reshape(1, -1))

        elif model.lower() == 'random_forest':
            model_ = joblib.load('model/random_forest')
            result = model_.predict(params_.reshape(1, -1))

        else:
            model_ = joblib.load('model/xgboost')
            result = model_.predict(params_.reshape(1, -1))

    else:
        model_ = joblib.load('model/xgboost')
        result = model_.predict(params_.reshape(1, -1))

    if result[0] == 0:
        status = 'Tidak Dapat Diminum'
        result_ = 0
    else:
        status = 'Dapat Diminum'
        result_ = 1

    return {
        'status': 200,
        'message': 'success',
        'method': 'post',
        'data': {
            'model': 'xgboost' if not model else model,
            'hasil': result_,
            'status': status
        }
    }
