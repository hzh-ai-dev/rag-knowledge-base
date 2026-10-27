"""向用户返回可操作的模型错误提示，避免暴露厂商原始响应。"""

from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError


def public_error(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return "模型认证失败，请检查 .env 中的 API Key 和接口地址。"
    if isinstance(exc, RateLimitError):
        return "模型服务限流或额度不足，请稍后重试并检查账户额度。"
    if isinstance(exc, APITimeoutError):
        return "模型响应超时，请稍后重试。"
    if isinstance(exc, APIConnectionError):
        return "无法连接模型服务，请检查网络和接口地址。"
    return "处理失败，请检查服务日志；确认模型名称、配置和资料格式后重试。"
