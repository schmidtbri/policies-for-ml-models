"""Policy decorator for MLModel instances."""
from typing import List, Union
from pydantic import BaseModel
from ml_base.decorator import MLModelDecorator

from opa_client.opa import OpaClient


class PredictionNotAvailable(BaseModel):
    """Schema returned when a prediction is not available because of a policy decision."""

    messages: List[str]


class OPAPolicyDecorator(MLModelDecorator):
    """Decorator to do policy checks using the Open Policy Agent service.

    Args:
        host: Hostname of the OPA service.
        port: Port of the OPA service.
        policy_package: Name of the policy to apply to the model.

    """

    def __init__(self, host: str, port: int, policy_package: str) -> None:
        """Initialize instance of OPAPolicyDecorator."""
        super().__init__(host=host, port=port, policy_package=policy_package)
        self.__dict__["_client"] = OpaClient(host=host,
                                             port=port,
                                             version="v1")

    @property
    def output_schema(self) -> BaseModel:
        """Decorate output_schema property to modify the model's output schema to accommodate the policy decision.

        Note:
            This method will create a Union of the model's output schema and the PredictionNotAvailable
            schema and return it.

        """

        class NewUnion(BaseModel):
            __root__: Union[self._model.output_schema, PredictionNotAvailable]

        NewUnion.__name__ = self._model.output_schema.__name__

        return NewUnion

    def predict(self, data):
        """Decorate the model's predict method, calling the OPA service with the model's input and output.

        Note:
            If a prediction is allowed the OPAPolicyDecorator predict() method will return an
            instance of the model's output schema. If a prediction is not allowed because of a policy
            violation, the decorator will return an instance of PredictionNotAvailable.

        """
        # make a prediction with the model
        prediction = self._model.predict(data=data)

        # build up data structure to send to the OPA service
        policy_check_data = {
            "model_qualified_name": self._model.qualified_name,
            "model_version": self._model.version,
            "model_input": data.dict(),
            "model_output": prediction.dict()
        }

        # call OPA service with model input and output
        response = self.__dict__["_client"].check_policy_rule(input_data=policy_check_data,
                                                              package_path=self._configuration["policy_package"])

        # if "allow" is True, then return the prediction
        if response["result"]["allow"]:
            return prediction
        # otherwise, return an instance of PredictionNotAvailable
        else:
            return PredictionNotAvailable(messages=response["result"]["messages"])

    def __del__(self):
        """Delete instance of OPAPolicyDecorator."""
        try:
            if self.__dict__["_client"] is not None:
                del self.__dict__["_client"]
        except KeyError:
            pass
