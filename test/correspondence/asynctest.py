
import asyncio
import subprocess


async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await asyncio.gather(*(sem_coro(c) for c in coros))

def run_sync(cmd):
    results = subprocess.run(cmd,capture_output=True,shell=True,text=True)
    return (results.returncode,results.stdout,results.stderr)

async def run_async_command(cmd):
    process = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    return (process.returncode,stdout.decode().strip(),stderr.decode().strip())

async def run_async_commands(commands):
    tasks = [asyncio.create_task(run_async_command(cmd)) for cmd in commands]
    # results = await asyncio.gather(*tasks)
    results = await gather_with_concurrency(3, *tasks)
    return results

# async def main():
#     commands = [f'sleep {i} && echo "sleep {i}"' for i in reversed(range(10))]
#     results = await run_async_commands(commands)
#     for result in results:
#         print(result)


# if __name__ == '__main__':
#     asyncio.run(main())

def main_():
    commands = [f'sleep {i} && echo "sleep {i}"' for i in reversed(range(10))]
    results = asyncio.run(run_async_commands(commands))
    for result in results:
        print(result)


if __name__ == '__main__':
    main_()