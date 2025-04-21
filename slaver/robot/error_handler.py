from tools.monitoring import (
    AgentLogger,
    LogLevel,
)


class ErrorHandler:
    def __init__(self, robot, error_definitions):
        self.robot = robot
        self.error_definitions = error_definitions
        self.logger = AgentLogger(level=LogLevel.INFO, log_file="./.log/agent.log")

    def find_error_info(self, error_code):
        """
        Search for the error definition based on the error code
        across all error categories.
        """
        for category in self.error_definitions.values():
            if error_code in category:
                return category[error_code]
        return None

    def handle_error(self, error_code):
        """
        Handle a given error by executing the recovery or resolution methods
        in the order defined in the error definitions.
        """
        error_info = self.find_error_info(error_code)
        if not error_info:
            self.logger.log(
                f"[ErrorHandler] Unknown error: {error_code}", LogLevel.INFO
            )
            return

        # Try to get one of the defined resolution strategy lists
        resolutions = (
            error_info.get("resolutionChecklist")
            or error_info.get("disambiguationProtocol")
            or error_info.get("resolution")
            or error_info.get("recoveryActions")
        )

        if not resolutions:
            self.logger.log(
                f"[ErrorHandler] {error_code} has no defined recovery methods",
                LogLevel.ERROR,
            )
            return

        self.logger.log(
            f"[ErrorHandler] Attempting recovery: {error_info['name']}", LogLevel.INFO
        )

        for method_name in resolutions.keys():
            try:
                # Remove trailing () if present and get the method from the robot
                method = getattr(self.robot, method_name.strip("()"))
                self.logger.log(
                    f"[Recovery] Executing recovery method: {method_name}",
                    LogLevel.INFO,
                )

                method()
                self.logger.log(
                    f"[Recovery] Method {method_name} executed successfully",
                    LogLevel.INFO,
                )
                return  # Stop after the first successful recovery
            except Exception as e:
                self.logger.log(
                    f"[Recovery] Method {method_name} failed: {e}", LogLevel.ERROR
                )
