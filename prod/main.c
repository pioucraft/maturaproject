#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <arpa/inet.h>

#include "strings.c"

#define PORT 8080
#define BUFFER_SIZE 4096

int server_fd, client_fd;
struct sockaddr_in addr;
socklen_t addr_len = sizeof(addr);
char buffer[BUFFER_SIZE];

void handle_signint(int sig) {
    printf("\nServer shutting down...\n");
    close(server_fd);
    exit(0);
}

char* response(char *status, char **headers, int header_count, char *body) {
    if(body == NULL) body = "";

    char *response_template = "HTTP/1.1 %s\r\n%s\r\n%s\r\n%s";
    int headers_length = 0;

    for(int i = 0; i< header_count; i++) {
        headers_length += strlen(headers[i]) + 2;
    }

    char *headers_combined = malloc(headers_length + 1);
    headers_combined[0] = '\0'; 

    for(int i = 0; i< header_count; i++) {
        strcat(headers_combined, headers[i]);
        strcat(headers_combined, "\r\n");
    }

    int body_length = strlen(body);
    char* content_length_header = malloc(30);
    sprintf(content_length_header, "Content-Length: %d", body_length);

    int response_length = strlen(response_template) - 8 // -8 for the %s
                          + strlen(status)
                          + strlen(content_length_header)
                          + headers_length
                          + strlen(body);
    char *response_str = malloc(response_length + 1);
    sprintf(response_str, response_template, status, content_length_header, headers_combined, body);

    free(headers_combined);
    free(content_length_header);

    return response_str;
}

char* return_html_file(char* filepath) {
    char buffer[256];

    FILE* file = fopen(filepath, "r");

    char* body = malloc(1);
    body[0] = '\0';

    while(fgets(buffer, sizeof(buffer), file)) {
        body = realloc(body, strlen(body) + strlen(buffer) + 1);
        strcat(body, buffer);
    }

    fclose(file);
    
    char* headers[] = {
        "Content-Type: text/html"
    };

    char* resp = response("200 OK", headers, 1, body);
    free(body);
    return resp;
}


char* web(char* method, char* path, char* version, char** headers, int headers_count, char* body) {
    return return_html_file("prod/index.html");
}

int main(int argc, char *argv[]) {
    signal(SIGINT, handle_signint);

    int option = 1;
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));

    if (server_fd < 0) {
        perror("Error creating socket");
        return 1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("Error binding socket");
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 500) < 0) {
        perror("Error listening on socket");
        close(server_fd);
        return 1;
    }

    printf("Server is listening on port %d\n", PORT);

    while (1) {
        client_fd = accept(server_fd, (struct sockaddr *)&addr, &addr_len);
        if (client_fd < 0) {
            perror("Error accepting connection");
            continue;
        }

        int total_bytes_received = 0;
        int finished = 0;
        char* currentBufferPtr = buffer;
        int content_length = 0;

        while(finished == 0) {
            int bytes_received = recv(client_fd, currentBufferPtr, BUFFER_SIZE - 1 - total_bytes_received, 0);
            currentBufferPtr += bytes_received;
            total_bytes_received += bytes_received;

            if (bytes_received < 0) {
                perror("Error receiving data");
                close(client_fd);
                finished = -1;
            }

            if(bytes_received == 0) {
                finished = 1;
                continue;
            }

            char* header_index = strstr(buffer, "Content-Length: ");
            if(header_index != NULL) {
                char* header_index_end = strstr(header_index, "\r\n");
                int header_length = header_index_end - header_index - 16; // minus 16 for strlen("Content-Length: ")

                char* header_value_string = malloc(header_length + 1);
                memcpy(header_value_string, header_index + 16, header_length); // + 16 because we only start at the end of "Content-Length: "
                header_value_string[header_length] = '\0';
                
                content_length = atoi(header_value_string);
                free(header_value_string);
            }

            char* headers_end_index = strstr(buffer, "\r\n\r\n");
            if(headers_end_index != NULL && content_length == 0) {
                finished = 1;
                continue;
            }

            if(headers_end_index != NULL) {
                int total_bytes_to_receive = (uint)(headers_end_index - buffer) + 4 + content_length; // +4 for the "\r\n\r\n"
                if(total_bytes_to_receive == total_bytes_received) {
                    finished = 1;
                    continue;
                }
            }

        } 

        if(finished == -1) {
            continue;
        }


        buffer[total_bytes_received] = '\0';

        char* headers = strtok_multi(buffer, "\r\n\r\n");
        char* body = strtok_multi(NULL, NULL);
        
        char** headers_slice = malloc(100 * sizeof(char*));
        int header_index = 0;
        char* line = strtok_multi(headers, "\r\n");
        while (strcmp(line, "") && header_index < 100) {
            headers_slice[header_index] = line;
            header_index++;
            line = strtok_multi(NULL, "\r\n");
        }
        
        char* method = strtok_multi(headers_slice[0], " ");
        char* path = strtok_multi(NULL, " ");
        char* version = strtok_multi(NULL, " ");
        
        headers_slice++;

        char* response = web(method, path, version, headers_slice, header_index, body);
        send(client_fd, response, strlen(response), 0);
        free(response);
        free(headers_slice - 1);
        

        close(client_fd);
    }

    close(server_fd);

    return 0;
}
