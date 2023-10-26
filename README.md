# Right Price Airbnb

Application to predict Airbnb prices.\
Home page allows users to input multiple features of their Airbnb listing.\
Result page returns a recommended price based on user selection.

**App:** https://rightpriceairbnb.azurewebsites.net/
<br><br>

### Home
<img src="https://user-images.githubusercontent.com/75267484/110042105-432ff880-7d13-11eb-81b8-7d63dc42c5c8.PNG" width="700"><br>

### Result
<img src="https://user-images.githubusercontent.com/75267484/110042045-25629380-7d13-11eb-9c1d-ec5041c99b5a.png" width="700"><br>
<br>

## Data & Tech
Built using Flask and deployed through Heroku. Backend coded in Python.\

**Dataset:** https://www.kaggle.com/rudymizrahi/airbnb-listings-in-major-us-cities-deloitte-ml

In order to offer the user the best experience of using the application, we transformed the amenities column into binary categories, accessed via checkboxes, in place of text input by the user. See the [Data_Exploration](https://github.com/JenFaith/airbnb_app/blob/main/Data_Exploration.ipynb) notebook for more details regarding the data preparation steps.  

We trained 3 different regression models (Linear Regression, Gradient Boosting Regressor, and Random Forest Regressor) and one [classification model](https://github.com/JenFaith/airbnb_app/blob/main/Classification_Model.ipynb) on the training data, utilizing cross-validation on the `train.csv` data.  
In agreement with the paper found [here](https://arxiv.org/ftp/arxiv/papers/1805/1805.12101.pdf), the model (`finalized_model.sav`) achieving the best score ($R^2=0.61$) in our experiments is a Random Forest Regressor of 200 trees each with a max depth of 20. Due to model file size considerations, we reduced the number of trees to 131 with max depth 10. The difference in model accuracy is negligible.

<br><br>

## Credits
### Meet the Creators

**Kevin Weatherwalks**<br>
https://www.linkedin.com/in/kevin-weatherwalks/<br>
https://github.com/KWeatherwalks <br>

**Stephen Lupsha**<br>
https://www.linkedin.com/in/stephen-lupsha-b60136140/<br>
https://github.com/StephenSpicer<br>

**Jennifer Faith**<br>
www.linkedin.com/in/jennifer-faith<br>
https://github.com/JenFaith<br>


**Filipe Collares**<br>
www.linkedin.com/in/fcollares<br>
https://github.com/fcollares<br>
<br>

## License
**MIT:** A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.<br>
MIT © [JenFaith]()
