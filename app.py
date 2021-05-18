from flask import Flask, render_template, request
import jsonify
import requests
import joblib
from tensorflow.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)
# load model and transformer
model = load_model('best_model.h5')
sc = joblib.load('scaler.pkl')

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        if (gender=='male'):
            gender=0
        else:
            gender=1
        senior = request.form['senior']
        if (senior == 'yes'):
            senior = 1
        else:
            senior = 0
        partner = request.form['partner']
        if (partner == 'yes'):
            partner = 1
        else:
            partner = 0
        dependents = request.form['dependents']
        if (dependents == 'yes'):
            dependents = 1
        else:
            dependents = 0
        tenure=int(request.form['tenure'])
        phoneserv = request.form['phoneserv']
        if (phoneserv == 'yes'):
            phoneserv = 1
        else:
            phoneserv = 0
        multilines = request.form['multilines']
        if (multilines == 'yes'):
            multilines = 1
        else:
            multilines = 0
        onlinesec = request.form['onlinesec']
        if (onlinesec == 'yes'):
            onlinesec = 1
        else:
            onlinesec = 0
        onlinebac = request.form['onlinebac']
        if (onlinebac == 'yes'):
            onlinebac = 1
        else:
            onlinebac = 0
        devicepro = request.form['devicepro']
        if (devicepro == 'yes'):
            devicepro = 1
        else:
            devicepro = 0
        techsup = request.form['techsup']
        if (techsup == 'yes'):
            techsup = 1
        else:
            techsup = 0
        strtv = request.form['strtv']
        if (strtv == 'yes'):
            strtv = 1
        else:
            strtv = 0
        strmvs = request.form['strmvs']
        if (strmvs == 'yes'):
            strmvs = 1
        else:
            strmvs = 0
        paperless = request.form['paperless']
        if (paperless == 'yes'):
            paperless = 1
        else:
            paperless = 0
        mb=float(request.form['mb'])
        tb=float(request.form['tb'])
        intserv_dsl=0
        intserv_fiber=0
        intserv_no=0
        intserv=request.form['intserv']
        if (intserv == 'dsl'):
            intserv_dsl = 1
        elif (intserv == 'fiber'):
            intserv_fiber=1
        else:
            intserv_no=1
        contract_mtom=0
        contract_oy=0
        contract_ty=0
        contract = request.form['contract']
        if (contract == 'mtom'):
            contract_mtom = 1
        elif (contract == 'oy'):
            contract_oy=1
        else:
            contract_ty=1
        pm_bt=0
        pm_cc=0
        pm_ec=0
        pm_mc=0
        pm=request.form['pm']
        if (pm == 'bt'):
            pm_bt=1
        elif (pm == 'cc'):
            pm_cc=1
        elif (pm == 'ec'):
            pm_ec=1
        else:
            pm_mc=1

        prediction=model.predict(sc.transform([[gender,senior,partner,dependents,tenure,phoneserv,multilines,onlinesec,
                                                        onlinebac,devicepro,techsup,strtv,strmvs,paperless,mb,tb,
                                                        intserv_dsl,intserv_fiber,intserv_no,
                                                        contract_mtom,contract_oy,contract_ty,
                                                        pm_bt,pm_cc,pm_ec,pm_mc]])) > 0.5
        output=prediction.tolist()

        if output==[[True]]:
            return render_template('index.html',prediction_text="Customer is unlikely to remain loyal")
        else:
            return render_template('index.html',prediction_text="Customer is likely to remain loyal")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

