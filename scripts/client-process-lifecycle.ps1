if ($null -eq ('EgoGlassProcessJob' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

public sealed class EgoGlassProcessJob : IDisposable
{
    private const uint JobObjectLimitKillOnJobClose = 0x00002000;
    private const int ExtendedLimitInformationClass = 9;
    private IntPtr handle;

    public EgoGlassProcessJob()
    {
        handle = CreateJobObject(IntPtr.Zero, null);
        if (handle == IntPtr.Zero)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }

        var limits = new JobObjectExtendedLimitInformation();
        limits.BasicLimitInformation.LimitFlags = JobObjectLimitKillOnJobClose;
        int length = Marshal.SizeOf(limits);
        IntPtr pointer = Marshal.AllocHGlobal(length);
        try
        {
            Marshal.StructureToPtr(limits, pointer, false);
            if (!SetInformationJobObject(
                handle,
                ExtendedLimitInformationClass,
                pointer,
                (uint)length))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
        }
        catch
        {
            CloseHandle(handle);
            handle = IntPtr.Zero;
            throw;
        }
        finally
        {
            Marshal.FreeHGlobal(pointer);
        }
    }

    public void AddProcess(int processId)
    {
        IntPtr currentHandle = handle;
        if (currentHandle == IntPtr.Zero)
        {
            throw new ObjectDisposedException("EgoGlassProcessJob");
        }

        using (Process process = Process.GetProcessById(processId))
        {
            bool isInJob;
            if (!IsProcessInJob(process.Handle, currentHandle, out isInJob))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            if (!isInJob && !AssignProcessToJobObject(currentHandle, process.Handle))
            {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
        }
    }

    public void Dispose()
    {
        IntPtr currentHandle = Interlocked.Exchange(ref handle, IntPtr.Zero);
        if (currentHandle != IntPtr.Zero)
        {
            CloseHandle(currentHandle);
        }
        GC.SuppressFinalize(this);
    }

    ~EgoGlassProcessJob()
    {
        Dispose();
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectBasicLimitInformation
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JobObjectExtendedLimitInformation
    {
        public JobObjectBasicLimitInformation BasicLimitInformation;
        public IoCounters IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr jobAttributes, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(
        IntPtr job,
        int informationClass,
        IntPtr information,
        uint informationLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool IsProcessInJob(
        IntPtr process,
        IntPtr job,
        out bool result);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);
}
'@
}

function New-EgoGlassProcessJob {
    return [EgoGlassProcessJob]::new()
}

function Add-ProcessTreeToJob {
    param(
        [Parameter(Mandatory)]
        [EgoGlassProcessJob] $Job,
        [Parameter(Mandatory)]
        [int] $ProcessId
    )

    try {
        $Job.AddProcess($ProcessId)
    } catch [System.ArgumentException] {
        if ($null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
            throw
        }
    }

    $children = @(
        Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" `
            -ErrorAction SilentlyContinue
    )
    foreach ($child in $children) {
        Add-ProcessTreeToJob -Job $Job -ProcessId ([int]$child.ProcessId)
    }
}

function Stop-ProcessTree {
    param(
        [Parameter(Mandatory)]
        [int] $ProcessId
    )

    $children = @(
        Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" `
            -ErrorAction SilentlyContinue
    )
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId ([int]$child.ProcessId)
    }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

function Stop-ClientProcesses {
    param(
        [AllowNull()]
        [object[]] $Processes,
        [AllowNull()]
        [EgoGlassProcessJob] $ProcessJob
    )

    try {
        foreach ($process in @($Processes)) {
            if ($null -ne $process) {
                Stop-ProcessTree -ProcessId ([int]$process.Id)
            }
        }
    } finally {
        if ($null -ne $ProcessJob) {
            $ProcessJob.Dispose()
        }
    }
}
