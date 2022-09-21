package insurance_charges_model
import future.keywords

test_customer_is_a_smoker_over_60 if {
    customer_is_a_smoker_over_60 with input as {
        "model_input": {
            "age": 62,
            "sex": "female",
            "bmi": 24.0,
            "children": 2,
            "smoker": true,
            "region": "northwest"
        },
        "model_output": {
            "charges": 12345.0
        }
    }
}

test_customer_is_a_smoker_over_60 if {
    not customer_is_a_smoker_over_60 with input as {
        "model_input": {
            "age": 42,
            "sex": "female",
            "bmi": 24.0,
            "children": 2,
            "smoker": true,
            "region": "northwest"
        },
        "model_output": {
            "charges": 12345.0
        }
    }
}

test_customer_is_a_smoker_over_60 if {
    not customer_is_a_smoker_over_60 with input as {
        "model_input": {
            "age": 62,
            "sex": "female",
            "bmi": 24.0,
            "children": 2,
            "smoker": false,
            "region": "northwest"
        },
        "model_output": {
            "charges": 12345.0
        }
    }
}

test_customer_is_a_smoker_over_60 if {
    not customer_is_a_smoker_over_60 with input as {
        "model_input": {
            "age": 42,
            "sex": "female",
            "bmi": 24.0,
            "children": 2,
            "smoker": false,
            "region": "northwest"
        },
        "model_output": {
            "charges": 12345.0
        }
    }
}

test_allow if {
    not allow with input as {
        "model_input": {
            "age": 62,
            "sex": "female",
            "bmi": 24.0,
            "children": 2,
            "smoker": true,
            "region": "northwest"
        },
        "model_output": {
            "charges": 12345.0
        }
    }
}

test_allow if {
    allow with input as {
        "model_input": {
            "age": 42,
            "sex": "female",
            "bmi": 24.0,
            "children": 2,
            "smoker": true,
            "region": "northwest"
        },
        "model_output": {
            "charges": 12345.0
        }
    }
}

test_allow if {
    allow with input as {
        "model_input": {
            "age": 62,
            "sex": "female",
            "bmi": 24.0,
            "children": 2,
            "smoker": false,
            "region": "northwest"
        },
        "model_output": {
            "charges": 12345.0
        }
    }
}

test_allow if {
    allow with input as {
        "model_input": {
            "age": 42,
            "sex": "female",
            "bmi": 24.0,
            "children": 2,
            "smoker": false,
            "region": "northwest"
        },
        "model_output": {
            "charges": 12345.0
        }
    }
}

contains_element(list, element) {
  list[_] = element
}

test_messages if {
    contains_element(messages, "Prediction cannot be made if customer is a smoker over the age of 60.") with input as {
        "model_input": {
            "age": 62,
            "sex": "female",
            "bmi": 24.0,
            "children": 2,
            "smoker": true,
            "region": "northwest"
        },
        "model_output": {
            "charges": 12345.0
        }
    }
}

test_messages if {
    not contains_element(messages, "Prediction cannot be made if customer is a smoker over the age of 60.") with input as {
        "model_input": {
            "age": 42,
            "sex": "female",
            "bmi": 24.0,
            "children": 2,
            "smoker": false,
            "region": "northwest"
        },
        "model_output": {
            "charges": 12345.0
        }
    }
}
