# Contributing Guidelines

## Merge Request Guidelines

TBD

## Branching Conventions

<table>
  <tbody>
    <tr>
      <th>Instance</th>
      <th align="center">Branch</th>
      <th align="center">Notes</th>
    </tr>
    <tr>
      <td>Official history</td>
      <td align="center">main</td>
      <td align="left">Stores the official history of the project. Accepts merges from develop and Hotfix.</td>
    </tr>
    <tr>
      <td>Working</td>
      <td align="center">develop</td>
      <td align="left">Serves as working (feature integration) branch. Accepts merges from feature, hotfix and bugfix and merges into
      a release branch.</td>
    </tr>
    <tr>
      <td>Release</td>
      <td align="center">release</td>
      <td align="left">Forks off develop branch and merges into main and tagged with a version. It should also be merged back into develop.</td>
    </tr>
    <tr>
      <td>Feature</td>
      <td align="center">feature</td>
      <td align="left">Always branches off and merge back into develop.</td>
    </tr>
    <tr>
      <td>Bugfix</td>
      <td align="center">bugfix</td>
      <td align="left">Always branches off and merge back into develop.</td>
    </tr>
    <tr>
      <td>Hotfix</td>
      <td align="center">hotfix</td>
      <td align="left">Always branches off and merges back into the main and the current release. It should also merge into develop.</td>
    </tr>
  </tbody>
</table>

### Main Branches

We hold three types of main evergreen branches: ``main``, ``develop`` and ``release``.

#### Main

The ``main`` branch should be considered ``origin/main`` and will be the main branch to store the official history
of the source code. The main branch always reflects the latest delivered development changes of the current release. All
releases are tagged and merged into the main branch.

#### Develoop


The ``develop`` branch should be considered ``develop/main`` and will be the working branch that serves as an integration
branch for features. As a developer, you will be branching off and merging back into ``develop``. The ``feature`` and
``bugfix`` branch off and merge back into ``develop``.

#### Release

The release branch always contain the latest code deployed. When the source code in the ``develop`` branch has enough
features, and is stable and ready to deploy, all the changes will be merged into a release branch tagged with a release
number. Release branches will eventually merged into main branch.

The release branch stores the source files for a specific version. We follow the `semantic versioning <https://semver.org/>`__
model to name release branches:

- The release branches should be named as ``release/*``.

- Release branch name follows the ``release/<X>.<Y>.x`` format where ``<X>``, ``<Y>`` are non-negative integers and
  ``x`` is string ``x``. Here, ``<X>`` represents the major version, ``<Y>`` represents the minor version, and the
  string ``x`` is an identifier for the all implemented patch versions.

- Patch versions do not create release branches, and they are only tagged with the complete release version number.
This project uses annotated tags for patch version tagging. Release branch tags follow ``r<X>.<Y>.<Z>`` format where same as
above ``<X>``, ``<Y>`` are non-negative integers representing the major and the minor versions. ``<Z>`` represents the
patch version.

- Any new new patches are directed towards ``main``, ``release`` and ``develop`` branches.

- Release branches are never deleted.

### Supporting Branches

We use three different types of supporting branches: ``feature``, ``bugfix`` and ``hotfix``.

Unlike the main branches, the supporting branches have a limited life time and are removed eventually when merged to
their target main branches.

### Feature

Any code changes for a new module or any new feature should be done on a
``feature`` branch. Feature branches are created based on the current
working branch (``develop``). When starting development, the deployment in
which this feature will be released may not be known. No matter when the
``feature`` branch will be finished, it will always be merged back into the
``develop`` branch. When all changes are done, you need to submit a merge
request to put all of these to the ``develop`` branch.

``feature`` branches has to be named as ``feature/``.\* and follow the
`Branch Naming <#branch-naming>`__

#### Bugfix

``bugfix`` branches are created when there is a bug on the latest
working branch that should be fixed and merged into the next release.
For that reason, a ``bugfix`` branch typically does not last longer than one
deployment cycle. A ``bugfix`` branch must be branched off from ``develop`` and
must merge back into ``develop``. ``bugfix`` branches are also used to
explicitly track the difference between ``bugfix`` development and ``feature``
development.

``bugfix`` branches has to be named as ``bugfix/``.\* and follow the
`Branch Naming <#branch-naming>`__

#### Hotfix

A ``hotfix`` branch comes from the need to act immediately upon an
unexpected issue of a stable ``release`` version (tag). Due to the
natural urgency, a ``hotfix`` is not required to be be pushed during a
scheduled deployment. A ``hotfix`` branch is branched off from a stable
``release`` tag and merged back to the ``main`` branch, and the stable
``release`` tag and the ``develop`` branch.

The ``hotfix`` branches has to be named as ``hotfix/``.\* and follow the
`Branch Naming <#branch-naming>`__

### Branch Naming

The branch names should follow the following pattern:

``<BRANCH-TYPE>/<ISSUE-NUMBER>_<SHORT-DESCRIPTION>``

The branch names should be all in lower caps letters, hyphen (-) should
be used to separate words and underscore (_) should be used to separate
the ID and description.
*Please note that th branch naming conventions are enforced via
GitLab*\ `push
request <https://docs.gitlab.com/ee/user/project/repository/push_rules.html>`__\ *and
any Git pushes not admitting to the naming convention as described above
will be rejected.*


## Commiting Conventions

To ensure that the commits are properly linked to their respective Jira tickets, the commit messages should either include the Jira ticket number enclosed in square brackets within the message like this: ``<COMMIT-MESSAGE>[DSCOE-XXXX]<COMMIT-MESSAGE>``, or they should begin with the Jira ticket number followed by an underscore and the commit message like this: ``DSCOE-XXXX_<COMMIT-MESSAGE>``. 