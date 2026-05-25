#include "SceneRTV2Context.h"

namespace SceneRTV2
{
    void FExportContext::AddIssue(const FString& Severity, const FString& Category,
                                  const FString& Message, const FString& RelatedId)
    {
        FValidationIssue Issue;
        Issue.Severity = Severity;
        Issue.Category = Category;
        Issue.Message = Message;
        Issue.RelatedId = RelatedId;
        Issues.Add(MoveTemp(Issue));
    }
}
