package insurance_charges_model

import future.keywords.contains
import future.keywords.if

customer_is_a_smoker_over_60 if {
	input.model_input.smoker
    input.model_input.age > 60
}

allow := true if {
	not customer_is_a_smoker_over_60
} else := false {
	customer_is_a_smoker_over_60
}

messages contains msg if {
	customer_is_a_smoker_over_60
    msg:= "Prediction cannot be made if customer is a smoker over the age of 60."
}
