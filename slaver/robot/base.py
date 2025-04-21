from abc import ABC, abstractmethod


class IMechanical(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def grasp(self, target):
        return {"status": "success", "message": "success"}

    @abstractmethod
    def navigate(self, target):
        return {"status": "success", "message": "success"}

    @abstractmethod
    def place(self, target):
        return {"status": "success", "message": "success"}
