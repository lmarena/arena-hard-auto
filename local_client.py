import sys
import json
import time
import uuid
import inspect
import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Type,
    Union,
    Generic,
    Mapping,
    TypeVar,
    Iterable,
    Iterator,
    Optional,
    Generator,
    AsyncIterator,
    cast,
    overload,
)
from typing_extensions import Literal, override, get_origin

DEFAULT_MAX_RETRIES = 2
DEFAULT_CONNECTION_LIMITS = 8
DEFAULT_TIMEOUT = 60

class LocalClient:
    _timeout = ...
    _model = ...

    def __init__(self):
        pass

    def request(self):
        pass

    def start(self):
        pass

    def close(self):
        pass

"""
class SyncAPIClient:
    _client: LocalClient
    _default_stream_cls: type[Stream[Any]] | None = None

    def __init__(
        self,
        *,
        version: str,
        base_url: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: float | None,
        client: LocalClient | None = None,
        custom_headers: Mapping[str, str] | None = None,
        custom_query: Mapping[str, object] | None = None,
        _strict_response_validation: bool,
    ) -> None:
        limits = DEFAULT_CONNECTION_LIMITS

        if not timeout:
            # if the user passed in a custom client with a non-default
            # timeout set then we use that timeout.
            #
            # note: there is an edge case here where the user passes in a client
            # where they've explicitly set the timeout to match the default timeout
            # as this check is structural, meaning that we'll think they didn't
            # pass in a timeout and will ignore it
            if client and client.timeout != DEFAULT_TIMEOUT:
                timeout = client.timeout
            else:
                timeout = DEFAULT_TIMEOUT

        if client is None or not isinstance(client, LocalClient):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"Invalid `client` argument; Expected an instance of `LocalClient` but got {type(client)}"
            )
        self._client = client
        self._version = version
        self._strict_response_validation = _strict_response_validation
        self.base_url = base_url
        self.max_retries = max_retries
        self.custom_headers = custom_headers
        self.custom_query = custom_query

    def is_closed(self) -> bool:
        return self._client.is_closed

    def close(self) -> None:
        #Close the underlying HTTPX client.
        #The client will *not* be usable after this.
        
        # If an error is thrown while constructing a client, self._client
        # may not be present
        if hasattr(self, "_client"):
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def _prepare_options(
        self,
        options,  # noqa: ARG002
    ):
        #Hook for mutating the given options
        return options

    def _prepare_request(
        self,
        request: Request,  # noqa: ARG002
    ) -> None:
        #This method is used as a callback for mutating the `Request` object
        #after it has been constructed.
        #This is useful for cases where you want to add certain headers based off of
        #the request properties, e.g. `url`, `method` etc.
        return None

    @overload
    def request(
        self,
        cast_to: Type[ResponseT],
        options,
        remaining_retries: Optional[int] = None,
        *,
    ) -> _StreamT: ...

    @overload
    def request(
        self,
        cast_to: Type[ResponseT],
        options,
        remaining_retries: Optional[int] = None,
        *,
    ) -> ResponseT: ...

    @overload
    def request(
        self,
        cast_to: Type[ResponseT],
        options,
        remaining_retries: Optional[int] = None,
        *,
        stream: bool = False,
    ):
        pass

    def request(
        self,
        cast_to: Type[ResponseT],
        options,
        remaining_retries: Optional[int] = None,
        *,
        stream: bool = False,
    ):
        if remaining_retries is not None:
            retries_taken = options.get_max_retries(self.max_retries) - remaining_retries
        else:
            retries_taken = 0

        return self._request(
            cast_to=cast_to,
            options=options,
            stream=stream,
            stream_cls=stream_cls,
            retries_taken=retries_taken,
        )

    def _request(
        self,
        *,
        cast_to: Type[ResponseT],
        options,
        retries_taken: int,
        stream: bool,
    ):
        # create a copy of the options we were given so that if the
        # options are mutated later & we then retry, the retries are
        # given the original options
        input_options = options.copy()

        cast_to = self._maybe_override_cast_to(cast_to, options)
        options = self._prepare_options(options)

        remaining_retries = options.get_max_retries(self.max_retries) - retries_taken
        request = self._build_request(options, retries_taken=retries_taken)
        self._prepare_request(request)

        kwargs: HttpxSendArgs = {}
        if self.custom_auth is not None:
            kwargs["auth"] = self.custom_auth

        log.debug("Sending HTTP Request: %s %s", request.method, request.url)

        try:
            response = self._client.send(
                request,
                **kwargs,
            )
        except httpx.TimeoutException as err:
            log.debug("Encountered httpx.TimeoutException", exc_info=True)

            if remaining_retries > 0:
                return self._retry_request(
                    input_options,
                    cast_to,
                    retries_taken=retries_taken,
                    response_headers=None,
                )

            log.debug("Raising timeout error")
            raise APITimeoutError(request=request) from err
        except Exception as err:
            log.debug("Encountered Exception", exc_info=True)

            if remaining_retries > 0:
                return self._retry_request(
                    input_options,
                    cast_to,
                    retries_taken=retries_taken,
                    response_headers=None,
                )

            log.debug("Raising connection error")
            raise APIConnectionError(request=request) from err

        log.debug(
            'HTTP Response: %s %s "%i %s" %s',
            request.method,
            request.url,
            response.status_code,
            response.reason_phrase,
            response.headers,
        )
        log.debug("request_id: %s", response.headers.get("x-request-id"))

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as err:  # thrown on 4xx and 5xx status code
            log.debug("Encountered httpx.HTTPStatusError", exc_info=True)

            if remaining_retries > 0 and self._should_retry(err.response):
                err.response.close()
                return self._retry_request(
                    input_options,
                    cast_to,
                    retries_taken=retries_taken,
                    response_headers=err.response.headers,
                    stream=stream,
                    stream_cls=stream_cls,
                )

            # If the response is streamed then we need to explicitly read the response
            # to completion before attempting to access the response text.
            if not err.response.is_closed:
                err.response.read()

            log.debug("Re-raising status error")
            raise self._make_status_error_from_response(err.response) from None

        return self._process_response(
            cast_to=cast_to,
            options=options,
            response=response,
            stream=stream,
            stream_cls=stream_cls,
            retries_taken=retries_taken,
        )

    def _retry_request(
        self,
        options,
        cast_to: Type[ResponseT],
        *,
        retries_taken: int,
        response_headers: httpx.Headers | None,
        stream: bool,
        stream_cls: type[_StreamT] | None,
    ) -> ResponseT | _StreamT:
        remaining_retries = options.get_max_retries(self.max_retries) - retries_taken
        if remaining_retries == 1:
            log.debug("1 retry left")
        else:
            log.debug("%i retries left", remaining_retries)

        timeout = self._calculate_retry_timeout(remaining_retries, options, response_headers)
        log.info("Retrying request to %s in %f seconds", options.url, timeout)

        # In a synchronous context we are blocking the entire thread. Up to the library user to run the client in a
        # different thread if necessary.
        time.sleep(timeout)

        return self._request(
            options=options,
            cast_to=cast_to,
            retries_taken=retries_taken + 1,
            stream=stream,
            stream_cls=stream_cls,
        )

    def _process_response(
        self,
        *,
        cast_to: Type[ResponseT],
        options,
        response: httpx.Response,
        stream: bool,
        stream_cls: type[Stream[Any]] | type[AsyncStream[Any]] | None,
        retries_taken: int = 0,
    ) -> ResponseT:
        if response.request.headers.get(RAW_RESPONSE_HEADER) == "true":
            return cast(
                ResponseT,
                LegacyAPIResponse(
                    raw=response,
                    client=self,
                    cast_to=cast_to,
                    stream=stream,
                    stream_cls=stream_cls,
                    options=options,
                    retries_taken=retries_taken,
                ),
            )

        origin = get_origin(cast_to) or cast_to

        if inspect.isclass(origin) and issubclass(origin, BaseAPIResponse):
            if not issubclass(origin, APIResponse):
                raise TypeError(f"API Response types must subclass {APIResponse}; Received {origin}")

            response_cls = cast("type[BaseAPIResponse[Any]]", cast_to)
            return cast(
                ResponseT,
                response_cls(
                    raw=response,
                    client=self,
                    cast_to=extract_response_type(response_cls),
                    stream=stream,
                    stream_cls=stream_cls,
                    options=options,
                    retries_taken=retries_taken,
                ),
            )

        if cast_to == httpx.Response:
            return cast(ResponseT, response)

        api_response = APIResponse(
            raw=response,
            client=self,
            cast_to=cast("type[ResponseT]", cast_to),  # pyright: ignore[reportUnnecessaryCast]
            stream=stream,
            stream_cls=stream_cls,
            options=options,
            retries_taken=retries_taken,
        )
        if bool(response.request.headers.get(RAW_RESPONSE_HEADER)):
            return cast(ResponseT, api_response)

        return api_response.parse()

    def _request_api_list(
        self,
        model: Type[object],
        page,
        options,
    ):
        pass

    def get(
        self,
        path: str,
        *,
        cast_to: Type[ResponseT],
        options: RequestOptions = {},
        stream: Literal[False] = False,
    ) -> ResponseT: ...

    def post(
        self,
        path: str,
        *,
        cast_to: Type[ResponseT],
        body: Body | None = None,
        options: RequestOptions = {},
        files: RequestFiles | None = None,
        stream: Literal[False] = False,
    ) -> ResponseT: ...

    def put(
        self,
        path: str,
        *,
        cast_to: Type[ResponseT],
        body: Body | None = None,
        files: RequestFiles | None = None,
        options: RequestOptions = {},
    ) -> ResponseT:
        opts = FinalRequestOptions.construct(
            method="put", url=path, json_data=body, files=to_httpx_files(files), **options
        )
        return self.request(cast_to, opts)
"""
