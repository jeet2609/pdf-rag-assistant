"""
Application logging configuration.

Provides a centralized logger configuration for the application.
"""

import logging

from config.settings import settings

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

# Log levels, from lowest to highest:
# DEBUG
# INFO
# WARNING
# ERROR
# CRITICAL

logging.basicConfig(
	level=settings.log_level,
	format=(
		"%(asctime)s | "
		"%(levelname)s | "
		"%(name)s | "
		"%(message)s"
	),
)

# --------------------------------------------------
# Logger Factory
# --------------------------------------------------


def get_logger(
	name: str,
) -> logging.Logger:
	"""
	Create or retrieve a logger for the given module name.

	Args:
		name:
			Name of the module requesting the logger.

	Returns:
		Configured logger instance.
	"""

	return logging.getLogger(name)